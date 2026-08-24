"""Privacy-preserving calibration feature bundle shared by edge and server."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np


BUNDLE_SCHEMA_VERSION = 1
PREPROCESS_NAME = "undistort-letterbox-rgb-v1"
DEFAULT_IMAGE_SIZE = 512
DEFAULT_CHUNK_BYTES = 512 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_feature_bundle(metadata: dict[str, Any], tokens: np.ndarray) -> bytes:
    """Encode metadata and FP16 tokens without executable object serialization."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 3 or tokens.shape[0] < 1:
        raise ValueError("tokens must have shape [camera, patch, embedding]")
    if not np.isfinite(tokens).all():
        raise ValueError("tokens contain non-finite values")
    document = dict(metadata)
    document["schema_version"] = BUNDLE_SCHEMA_VERSION
    document["token_shape"] = list(tokens.shape)
    document["token_dtype"] = "float16"
    raw_metadata = json.dumps(
        document, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    destination = BytesIO()
    np.savez_compressed(
        destination,
        metadata=np.frombuffer(raw_metadata, dtype=np.uint8),
        tokens=tokens.astype(np.float16, copy=False),
    )
    payload = destination.getvalue()
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ValueError("feature bundle exceeds the size limit")
    return payload


def decode_feature_bundle(payload: bytes) -> tuple[dict[str, Any], np.ndarray]:
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ValueError("feature bundle exceeds the size limit")
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive_file:
            members = archive_file.infolist()
            if {member.filename for member in members} != {"metadata.npy", "tokens.npy"}:
                raise ValueError("feature bundle contains unexpected fields")
            if sum(member.file_size for member in members) > MAX_BUNDLE_BYTES:
                raise ValueError("expanded feature bundle exceeds the size limit")
    except zipfile.BadZipFile as exc:
        raise ValueError("feature bundle is not a valid NPZ archive") from exc
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {"metadata", "tokens"}:
            raise ValueError("feature bundle contains unexpected fields")
        raw_metadata = archive["metadata"]
        tokens = archive["tokens"]
    if raw_metadata.dtype != np.uint8 or raw_metadata.ndim != 1:
        raise ValueError("invalid feature metadata encoding")
    metadata = json.loads(raw_metadata.tobytes().decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("feature metadata must be an object")
    if metadata.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported feature bundle schema")
    expected_shape = tuple(int(value) for value in metadata.get("token_shape", ()))
    if tokens.dtype != np.float16 or tokens.ndim != 3 or tokens.shape != expected_shape:
        raise ValueError("feature token dtype or shape does not match metadata")
    if tokens.shape[0] > 256 or tokens.shape[1] > 4096 or tokens.shape[2] > 4096:
        raise ValueError("feature token dimensions exceed the limit")
    if not np.isfinite(tokens).all():
        raise ValueError("feature tokens contain non-finite values")
    cameras = metadata.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != tokens.shape[0]:
        raise ValueError("camera metadata and token count do not match")
    camera_keys = [camera.get("camera_key") for camera in cameras if isinstance(camera, dict)]
    if len(camera_keys) != len(cameras) or any(not key for key in camera_keys):
        raise ValueError("each camera must have a camera_key")
    if len(set(camera_keys)) != len(camera_keys):
        raise ValueError("camera_key values must be unique")
    return metadata, tokens


def split_payload(payload: bytes, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> list[bytes]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    return [payload[offset : offset + chunk_bytes] for offset in range(0, len(payload), chunk_bytes)]


@dataclass
class ChunkCollector:
    expected_count: int
    expected_sha256: str
    expected_size: int

    def __post_init__(self) -> None:
        if self.expected_count <= 0 or self.expected_size <= 0:
            raise ValueError("invalid transfer dimensions")
        if self.expected_size > MAX_BUNDLE_BYTES:
            raise ValueError("feature bundle exceeds the size limit")
        self._chunks: dict[int, bytes] = {}

    def add(self, index: int, payload: bytes) -> None:
        if index < 0 or index >= self.expected_count:
            raise ValueError("chunk index is out of range")
        self._chunks.setdefault(index, bytes(payload))

    @property
    def complete(self) -> bool:
        return len(self._chunks) == self.expected_count

    def assemble(self) -> bytes:
        if not self.complete:
            raise ValueError("feature transfer is incomplete")
        payload = b"".join(self._chunks[index] for index in range(self.expected_count))
        if len(payload) != self.expected_size:
            raise ValueError("feature transfer size mismatch")
        if hashlib.sha256(payload).hexdigest() != self.expected_sha256:
            raise ValueError("feature transfer checksum mismatch")
        return payload


def payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
