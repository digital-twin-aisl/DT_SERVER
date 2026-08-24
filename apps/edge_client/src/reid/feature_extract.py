# ruff: noqa: E402

import collections
from collections.abc import Mapping
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import numpy as np

if "bool" not in np.__dict__:
    np.bool = np.bool_

import torch
from PIL import Image

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fastreid_dir = os.path.join(base_dir, "reid", "fast-reid")
sys.path.append(fastreid_dir)

# The pinned FastReID revision predates Python 3.10, where these aliases moved
# to collections.abc. Keep the compatibility shim local to the ReID adapter.
if not hasattr(collections, "Mapping"):
    setattr(collections, "Mapping", Mapping)

from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.transforms import build_transforms
from apps.edge_client.src.utils.tensorrt import (
    _checkpoint_sha256,
    _convert_component,
    _hardware_identity,
    _spec_fingerprint,
)


alpha = 1.0
REID_ENGINE_CACHE_SCHEMA = 2
REID_ENGINE_FILE = "model.pth"
REID_MIN_BATCH_SIZE = 1
REID_OPT_BATCH_SIZE = 4
REID_MAX_BATCH_SIZE = 16


def _reid_config():
    cfg_path = Path(fastreid_dir) / "configs" / "Market1501" / "bagtricks_R50-ibn.yml"
    checkpoint = Path(fastreid_dir) / "weights" / "market_bot_R50-ibn.pth"
    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_path))
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.BACKBONE.PRETRAIN = False
    return cfg, checkpoint


# ======================================================================
# 기본 PyTorch 기반 Feature Extractor
# ======================================================================

def build_feature_extractor():
    """
    FastReID PyTorch 모델과 전처리 transform을 생성한다.
    """
    cfg, checkpoint = _reid_config()

    model = build_model(cfg)
    model.eval()
    Checkpointer(model).load(str(checkpoint))

    transform = build_transforms(cfg, is_train=False)
    return model, transform


@torch.no_grad()
def extract_features_from_persons(model, transform, person_list):
    """
    PyTorch FastReID 모델을 사용해 person_list에서 feature를 추출한다.
    Procrustes distance 기반 가중치를 feature에 곱해 반환한다.
    """
    features = []
    weights = _person_weights(person_list)

    for i, person in enumerate(person_list):
        crop_img = person["crop"]

        crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        img_tensor = transform(crop_pil).unsqueeze(0).to(next(model.parameters()).device)

        output = model(img_tensor)
        feat = output.cpu().numpy().squeeze(0)

        weight = weights[i]
        feat_weighted = feat * weight

        features.append(
            {
                "edge_id": person["edge_id"],
                "cam": person["cam"],
                "frame": person["frame"],
                "person_idx": person["person_idx"],
                "bbox": [float(x) for x in person["bbox"]],
                "feature": feat_weighted.astype(np.float32, copy=False),
                "weight": float(weight),
            }
        )

    return features


def procrustes_distance(A, B):
    """
    두 keypoint 세트 A, B 사이의 Procrustes distance를 계산한다.
    실패 시 np.nan을 반환한다.
    """
    try:
        A_mean, B_mean = A.mean(axis=0), B.mean(axis=0)
        A_centered, B_centered = A - A_mean, B - B_mean

        norm_A, norm_B = np.linalg.norm(A_centered), np.linalg.norm(B_centered)
        A_scaled, B_scaled = A_centered / norm_A, B_centered / norm_B

        U, _, Vt = np.linalg.svd(B_scaled.T @ A_scaled)
        R = U @ Vt
        A_aligned = A_scaled @ R.T

        return np.sqrt(((A_aligned - B_scaled) ** 2).sum())
    except Exception:
        return np.nan


def _person_weights(person_list):
    """Keep Procrustes weighting local to each camera after GPU batching."""
    weights = []
    for index, person in enumerate(person_list):
        distances = [
            procrustes_distance(person["keypoints"], other["keypoints"])
            for other_index, other in enumerate(person_list)
            if other_index != index and other["cam"] == person["cam"]
        ]
        distances = [value for value in distances if not np.isnan(value)]
        weights.append(
            float(1 + alpha * np.mean(distances)) if distances else 1.0
        )
    return weights


# ======================================================================
# TensorRT-based feature extractor with target-specific atomic caching
# ======================================================================


class _ReIDInferenceModule(torch.nn.Module):
    """Protect the caller input from FastReID's in-place normalization."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images.clone())


def build_reid_engine_spec(checkpoint, cfg, mode="fp16"):
    """Return every input/configuration value affecting the ReID engine."""
    checkpoint = Path(checkpoint).resolve()
    config_sha256 = hashlib.sha256(cfg.dump().encode("utf-8")).hexdigest()
    return {
        "schema": REID_ENGINE_CACHE_SCHEMA,
        "checkpoint": {
            "name": checkpoint.name,
            "sha256": _checkpoint_sha256(checkpoint),
        },
        "config_sha256": config_sha256,
        "precision": mode,
        "input_shape": [
            REID_MIN_BATCH_SIZE,
            3,
            int(cfg.INPUT.SIZE_TEST[0]),
            int(cfg.INPUT.SIZE_TEST[1]),
        ],
        "batch_range": [
            REID_MIN_BATCH_SIZE,
            REID_OPT_BATCH_SIZE,
            REID_MAX_BATCH_SIZE,
        ],
        "output_features": int(cfg.MODEL.BACKBONE.FEAT_DIM),
        "hardware": _hardware_identity(),
    }


def _reid_cache_paths(checkpoint, spec):
    checkpoint = Path(checkpoint).resolve()
    edge_client_root = Path(__file__).resolve().parents[2]
    cache_root = edge_client_root / "models" / f"{checkpoint.name}.tensorrt"
    return cache_root, cache_root / _spec_fingerprint(spec)


def _reid_cache_is_valid(engine_dir, spec):
    try:
        manifest = json.loads(
            (engine_dir / "manifest.json").read_text(encoding="utf-8")
        )
        return (
            manifest == spec
            and (engine_dir / REID_ENGINE_FILE).stat().st_size > 0
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _remove_stale_reid_builds(cache_root, engine_dir):
    for path in cache_root.glob(f".{engine_dir.name}.*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _build_reid_cache(cache_root, engine_dir, spec, mode):
    model, _ = build_feature_extractor()
    model = _ReIDInferenceModule(model).eval().cuda()
    _, opt_batch_size, max_batch_size = spec["batch_range"]
    example_shape = list(spec["input_shape"])
    example_shape[0] = opt_batch_size
    example = torch.rand(
        tuple(example_shape),
        device="cuda",
        dtype=torch.float32,
    ) * 255.0
    min_shape = tuple(spec["input_shape"])
    max_shape = (max_batch_size, *min_shape[1:])
    converted = _convert_component(
        "fastreid",
        model,
        example,
        mode,
        min_shapes=[min_shape],
        opt_shapes=[tuple(example_shape)],
        max_shapes=[max_shape],
    )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{engine_dir.name}.", dir=cache_root)
    )
    try:
        torch.save(converted.state_dict(), temporary / REID_ENGINE_FILE)
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
    finally:
        del converted
        del model
        torch.cuda.empty_cache()


def build_trt_feature_extractor(force_rebuild=False, mode="fp16"):
    """Build the target-specific ReID engine once, then load its TRTModule."""
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT ReID inference requires CUDA")

    from torch2trt import TRTModule

    cfg, checkpoint = _reid_config()
    spec = build_reid_engine_spec(checkpoint, cfg, mode)
    cache_root, engine_dir = _reid_cache_paths(checkpoint, spec)
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / ".build.lock"

    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _remove_stale_reid_builds(cache_root, engine_dir)
        cache_valid = _reid_cache_is_valid(engine_dir, spec)
        if force_rebuild or not cache_valid:
            action = "Rebuilding" if force_rebuild else "Building"
            print(f"{action} TensorRT ReID engine: {engine_dir}")
            _build_reid_cache(cache_root, engine_dir, spec, mode)
        else:
            print(f"Reusing TensorRT ReID engine: {engine_dir}")

    module = TRTModule()
    module.load_state_dict(
        torch.load(
            engine_dir / REID_ENGINE_FILE,
            map_location="cuda",
            weights_only=False,
        )
    )
    transform = build_transforms(cfg, is_train=False)
    print(f"TensorRT ReID model ready: precision={mode}")
    return module.eval(), transform


@torch.no_grad()
def extract_features_from_persons_trt(model, transform, person_list):
    """Extract ReID features directly with the cached CUDA TRTModule."""
    if not person_list:
        return []
    weights = _person_weights(person_list)
    tensors = []
    for person in person_list:
        crop_pil = Image.fromarray(
            cv2.cvtColor(person["crop"], cv2.COLOR_BGR2RGB)
        )
        tensors.append(transform(crop_pil))

    feature_batches = []
    for start in range(0, len(tensors), REID_MAX_BATCH_SIZE):
        image_batch = torch.stack(
            tensors[start : start + REID_MAX_BATCH_SIZE]
        ).contiguous().cuda(non_blocking=True)
        feature_batches.append(model(image_batch).float().cpu())
    feature_array = torch.cat(feature_batches).numpy()

    features = []
    for index, (person, feature, weight) in enumerate(
        zip(person_list, feature_array, weights)
    ):
        features.append(
            {
                "edge_id": person["edge_id"],
                "cam": person["cam"],
                "frame": person["frame"],
                "person_idx": person["person_idx"],
                "bbox": [float(value) for value in person["bbox"]],
                "feature": (feature * weight).astype(np.float32, copy=False),
                "weight": float(weight),
            }
        )
    return features
