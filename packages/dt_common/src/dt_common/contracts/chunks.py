# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Bounded binary transfer assembly; independent of NumPy and transport SDKs."""

from dataclasses import dataclass
import hashlib

MAX_BUNDLE_BYTES = 256 * 1024 * 1024


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

