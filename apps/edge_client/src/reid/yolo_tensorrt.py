"""Target-specific TensorRT caching for the edge YOLO pose detector."""

from __future__ import annotations

import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile

from apps.edge_client.src.utils.tensorrt import (
    _checkpoint_sha256,
    _hardware_identity,
    _spec_fingerprint,
)


YOLO_ENGINE_CACHE_SCHEMA = 1
YOLO_ENGINE_FILE = "model.engine"


def build_yolo_engine_spec(checkpoint, *, batch_size, image_size, mode="fp16"):
    checkpoint = Path(checkpoint).resolve()
    return {
        "schema": YOLO_ENGINE_CACHE_SCHEMA,
        "checkpoint": {
            "name": checkpoint.name,
            "sha256": _checkpoint_sha256(checkpoint),
        },
        "ultralytics": version("ultralytics"),
        "precision": mode,
        "batch_size": int(batch_size),
        "image_size": int(image_size),
        "hardware": _hardware_identity(),
    }


def _cache_paths(checkpoint, spec):
    checkpoint = Path(checkpoint).resolve()
    cache_root = checkpoint.parent / f"{checkpoint.name}.tensorrt"
    return cache_root, cache_root / _spec_fingerprint(spec)


def _cache_is_valid(engine_dir, spec):
    try:
        manifest = json.loads(
            (engine_dir / "manifest.json").read_text(encoding="utf-8")
        )
        return (
            manifest == spec
            and (engine_dir / YOLO_ENGINE_FILE).stat().st_size > 0
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _remove_stale_builds(cache_root, engine_dir):
    for path in cache_root.glob(f".{engine_dir.name}.*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _build_cache(checkpoint, cache_root, engine_dir, spec):
    from ultralytics import YOLO

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{engine_dir.name}.", dir=cache_root)
    )
    try:
        temporary_checkpoint = temporary / checkpoint.name
        shutil.copy2(checkpoint, temporary_checkpoint)
        exported = Path(
            YOLO(str(temporary_checkpoint)).export(
                format="engine",
                half=spec["precision"] == "fp16",
                batch=spec["batch_size"],
                imgsz=spec["image_size"],
                device=0,
                dynamic=False,
                simplify=False,
                opset=18,
                verbose=False,
            )
        )
        if not exported.is_file() or exported.stat().st_size <= 0:
            raise RuntimeError("Ultralytics did not produce a YOLO engine")
        exported.replace(temporary / YOLO_ENGINE_FILE)
        for build_artifact in temporary.iterdir():
            if build_artifact.name in {YOLO_ENGINE_FILE, "manifest.json"}:
                continue
            if build_artifact.is_dir():
                shutil.rmtree(build_artifact)
            else:
                build_artifact.unlink()
        (temporary / "manifest.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if engine_dir.exists():
            shutil.rmtree(engine_dir)
        os.replace(temporary, engine_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_trt_yolo(checkpoint, *, batch_size, image_size, force_rebuild=False):
    """Build the fixed four-camera engine once and return an Ultralytics model."""
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).resolve()
    spec = build_yolo_engine_spec(
        checkpoint,
        batch_size=batch_size,
        image_size=image_size,
    )
    cache_root, engine_dir = _cache_paths(checkpoint, spec)
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / ".build.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _remove_stale_builds(cache_root, engine_dir)
        if force_rebuild or not _cache_is_valid(engine_dir, spec):
            action = "Rebuilding" if force_rebuild else "Building"
            print(f"{action} TensorRT YOLO pose engine: {engine_dir}")
            _build_cache(checkpoint, cache_root, engine_dir, spec)
        else:
            print(f"Reusing TensorRT YOLO pose engine: {engine_dir}")
    print(f"TensorRT YOLO pose model ready: precision=fp16 batch={batch_size}")
    return YOLO(str(engine_dir / YOLO_ENGINE_FILE), task="pose")
