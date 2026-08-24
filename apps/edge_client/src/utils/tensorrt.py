"""TensorRT engine creation and caching for the edge pose model."""

from __future__ import annotations

import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from unittest.mock import patch

import torch


ENGINE_CACHE_SCHEMA = 1
ENGINE_FILES = {
    "backbone": "backbone.pth",
    "root_v2v_net": "root_v2v_net.pth",
    "pose_v2v_net": "pose_v2v_net.pth",
}
DEFAULT_WORKSPACE_SIZE = 1 << 30


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hardware_identity() -> dict[str, Any]:
    import tensorrt

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    try:
        torch2trt_version = version("torch2trt")
    except PackageNotFoundError:
        torch2trt_version = "unknown"
    return {
        "torch": torch.__version__,
        "torch2trt": torch2trt_version,
        "tensorrt": tensorrt.__version__,
        "cuda": torch.version.cuda,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
    }


def build_engine_spec(model, checkpoint_path, cfg, mode="fp16"):
    """Return every value that affects the serialized TensorRT engines."""
    checkpoint = Path(checkpoint_path).resolve()
    include_pose = getattr(model, "inference_mode", None) != "rootnet"
    spec = {
        "schema": ENGINE_CACHE_SCHEMA,
        "checkpoint": {
            "name": checkpoint.name,
            "sha256": _checkpoint_sha256(checkpoint),
        },
        "precision": mode,
        "batch_size": int(cfg.BATCH_SIZE),
        "backbone_batch_size": int(cfg.BATCH_SIZE) * int(cfg.NUM_VIEWS),
        "num_views": int(cfg.NUM_VIEWS),
        "image_size": [int(value) for value in cfg.NETWORK.IMAGE_SIZE],
        "num_joints": int(cfg.NETWORK.NUM_JOINTS),
        "root_cube_size": [int(value) for value in model.root_net.cube_size],
        "pose_cube_size": [int(value) for value in cfg.PICT_STRUCT.CUBE_SIZE],
        "inference_mode": getattr(model, "inference_mode", None),
        "components": [
            "backbone",
            "root_v2v_net",
            *(["pose_v2v_net"] if include_pose else []),
        ],
        "hardware": _hardware_identity(),
    }
    return spec


def _spec_fingerprint(spec):
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _cache_paths(checkpoint_path, spec):
    checkpoint = Path(checkpoint_path).resolve()
    cache_root = checkpoint.parent / f"{checkpoint.name}.tensorrt"
    engine_dir = cache_root / _spec_fingerprint(spec)
    return cache_root, engine_dir


def _read_manifest(engine_dir):
    try:
        return json.loads(
            (engine_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _cache_is_valid(engine_dir, spec):
    if _read_manifest(engine_dir) != spec:
        return False
    try:
        return all(
            (engine_dir / ENGINE_FILES[component]).stat().st_size > 0
            for component in spec["components"]
        )
    except OSError:
        return False


def _convert_component(
    name,
    module,
    example,
    mode,
    *,
    min_shapes=None,
    opt_shapes=None,
    max_shapes=None,
):
    from torch2trt import torch2trt

    if mode != "fp16":
        raise ValueError(f"unsupported TensorRT precision: {mode}")

    print(f"Building TensorRT component: {name} input={tuple(example.shape)}")
    module = module.eval()

    # torch2trt 0.5 passes legacy dynamic_axes to torch.onnx.export. PyTorch
    # 2.11 defaults to the dynamo exporter, where that legacy argument has a
    # different input-tree contract. Force the stable TorchScript exporter for
    # this fixed-shape engine conversion.
    torch_onnx_export = torch.onnx.export

    def legacy_onnx_export(*args, **kwargs):
        args[0].eval()
        kwargs["dynamo"] = False
        kwargs["training"] = torch.onnx.TrainingMode.EVAL
        return torch_onnx_export(*args, **kwargs)

    with torch.inference_mode():
        reference = module(example)
        with patch("torch.onnx.export", legacy_onnx_export):
            converted = torch2trt(
                module,
                [example],
                fp16_mode=True,
                use_onnx=True,
                max_workspace_size=DEFAULT_WORKSPACE_SIZE,
                min_shapes=min_shapes,
                opt_shapes=opt_shapes,
                max_shapes=max_shapes,
            )
        actual = converted(example)
        torch.cuda.synchronize()

    if reference.shape != actual.shape:
        raise RuntimeError(
            f"TensorRT {name} output shape mismatch: "
            f"PyTorch={tuple(reference.shape)}, TensorRT={tuple(actual.shape)}"
        )
    if not torch.isfinite(actual).all():
        raise RuntimeError(f"TensorRT {name} produced non-finite output")

    difference = (reference.float() - actual.float()).abs()
    print(
        f"Validated TensorRT component: {name} "
        f"max_abs_error={difference.max().item():.6f} "
        f"mean_abs_error={difference.mean().item():.6f}"
    )
    return converted


def export_tensorrt(model, output_dir, cfg, spec, mode="fp16"):
    """Build the components used by this model and write a complete cache."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = next(model.parameters()).device
    batch_size = int(cfg.BATCH_SIZE)
    backbone_batch_size = batch_size * int(cfg.NUM_VIEWS)
    image_width, image_height = (int(value) for value in cfg.NETWORK.IMAGE_SIZE)
    root_cube_size = tuple(int(value) for value in model.root_net.cube_size)

    examples = {
        "backbone": torch.ones(
            (backbone_batch_size, 3, image_height, image_width),
            device=device,
        ),
        "root_v2v_net": torch.ones(
            (batch_size, 1, *root_cube_size),
            device=device,
        ),
    }
    modules = {
        "backbone": model.backbone,
        "root_v2v_net": model.root_net.v2v_net,
    }
    if "pose_v2v_net" in spec["components"]:
        pose_cube_size = tuple(int(value) for value in cfg.PICT_STRUCT.CUBE_SIZE)
        examples["pose_v2v_net"] = torch.ones(
            (1, int(cfg.NETWORK.NUM_JOINTS), *pose_cube_size),
            device=device,
        )
        modules["pose_v2v_net"] = model.pose_net.v2v_net

    for component in spec["components"]:
        converted = _convert_component(
            component,
            modules[component],
            examples[component],
            mode,
        )
        torch.save(converted.state_dict(), output / ENGINE_FILES[component])
        del converted
        torch.cuda.empty_cache()

    (output / "manifest.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_cache(model, cache_root, engine_dir, cfg, spec, mode):
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{engine_dir.name}.", dir=cache_root)
    )
    # export_tensorrt creates the output directory itself.
    temporary.rmdir()
    try:
        export_tensorrt(model, temporary, cfg, spec, mode)
        if engine_dir.exists():
            shutil.rmtree(engine_dir)
        os.replace(temporary, engine_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _remove_stale_builds(cache_root, engine_dir):
    """Remove interrupted temporary builds for this exact engine fingerprint."""
    for path in cache_root.glob(f".{engine_dir.name}.*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _load_component(engine_dir, component, device):
    from torch2trt import TRTModule

    module = TRTModule()
    state = torch.load(
        engine_dir / ENGINE_FILES[component],
        map_location=device,
        weights_only=False,
    )
    module.load_state_dict(state)
    return module.eval()


def load_tensorrt_model(
    model,
    checkpoint_path,
    cfg,
    mode="fp16",
    force_rebuild=False,
):
    """Build a configuration-specific engine once, then load it on each run."""
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT inference requires CUDA")

    spec = build_engine_spec(model, checkpoint_path, cfg, mode)
    cache_root, engine_dir = _cache_paths(checkpoint_path, spec)
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / ".build.lock"

    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _remove_stale_builds(cache_root, engine_dir)
        cache_valid = _cache_is_valid(engine_dir, spec)
        if force_rebuild or not cache_valid:
            action = "Rebuilding" if force_rebuild else "Building"
            print(
                f"{action} TensorRT pose engine for the current checkpoint, "
                f"configuration, and device: {engine_dir}"
            )
            _build_cache(model, cache_root, engine_dir, cfg, spec, mode)
        else:
            print(f"Reusing TensorRT pose engine: {engine_dir}")

    device = next(model.parameters()).device
    model.backbone = _load_component(engine_dir, "backbone", device)
    model.root_net.v2v_net = _load_component(
        engine_dir,
        "root_v2v_net",
        device,
    )
    if "pose_v2v_net" in spec["components"]:
        model.pose_net.v2v_net = _load_component(
            engine_dir,
            "pose_v2v_net",
            device,
        )
    print(f"TensorRT pose model ready: precision={mode}")
    return model
