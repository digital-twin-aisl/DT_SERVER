"""Versioned, bounded inference messages shared by edge, server and replay.

ZNH2 carries zstd-compressed JSON metadata followed by contiguous tensor bytes.
No executable Python objects are deserialized. ZNH1/pickle is deliberately not
accepted: deploy the edge and server together when upgrading the protocol.
"""

import json
import math
import struct

import numpy as np
import zstandard as zstd


MAGIC = b"ZNH2"
HEADER = struct.Struct("!4sdI")
LENGTH = struct.Struct("!I")
MAX_MESSAGE_BYTES = 32 * 1024 * 1024
MAX_RAW_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_ARRAYS = 2048
DTYPES = {"|u1", "<f4", "<f8", "<i4", "<i8"}


def inspect_message(payload):
    if len(payload) > MAX_MESSAGE_BYTES or len(payload) < HEADER.size:
        raise ValueError("invalid inference message size")
    magic, timestamp, raw_size = HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError("expected ZNH2; upgrade edge/server together")
    if not math.isfinite(timestamp) or not 0 < raw_size <= MAX_RAW_BYTES:
        raise ValueError("invalid inference header")
    return timestamp, raw_size


def encode_frame(frame):
    arrays = []
    descriptors = []
    offset = 0

    def encode(value):
        nonlocal offset
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            if array.dtype.str not in DTYPES or array.ndim > 5:
                raise ValueError("unsupported inference tensor dtype or rank")
            if len(arrays) >= MAX_ARRAYS:
                raise ValueError("too many inference tensors")
            index = len(arrays)
            arrays.append(array.tobytes())
            descriptors.append(
                {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "offset": offset,
                    "size": array.nbytes,
                }
            )
            offset += array.nbytes
            if offset > MAX_RAW_BYTES:
                raise ValueError("inference tensors exceed size limit")
            return {"__tensor__": index}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            if "__tensor__" in value:
                raise ValueError("reserved tensor metadata key")
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        return value

    metadata = json.dumps(
        {"schema_version": 2, "frame": encode(frame), "tensors": descriptors},
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if len(metadata) > MAX_METADATA_BYTES:
        raise ValueError("inference metadata exceeds size limit")
    raw = LENGTH.pack(len(metadata)) + metadata + b"".join(arrays)
    if len(raw) > MAX_RAW_BYTES:
        raise ValueError("inference frame exceeds size limit")
    timestamp = float(frame["time"])
    if not math.isfinite(timestamp):
        raise ValueError("timestamp must be finite")
    message = HEADER.pack(MAGIC, timestamp, len(raw)) + zstd.ZstdCompressor(
        level=1, write_checksum=True
    ).compress(raw)
    inspect_message(message)
    return message


def decode_frame(payload):
    timestamp, raw_size = inspect_message(payload)
    compressed = payload[HEADER.size :]
    declared = zstd.frame_content_size(compressed)
    if declared != raw_size:
        raise ValueError("compressed frame size does not match header")
    raw = zstd.ZstdDecompressor().decompress(compressed, max_output_size=raw_size)
    if len(raw) != raw_size or len(raw) < LENGTH.size:
        raise ValueError("invalid decoded frame size")
    (metadata_size,) = LENGTH.unpack_from(raw)
    if not 0 < metadata_size <= min(MAX_METADATA_BYTES, len(raw) - LENGTH.size):
        raise ValueError("invalid metadata size")
    metadata = json.loads(raw[LENGTH.size : LENGTH.size + metadata_size])
    if metadata.get("schema_version") != 2:
        raise ValueError("unsupported inference schema")
    binary = memoryview(raw)[LENGTH.size + metadata_size :]
    arrays = []
    offset = 0
    descriptors = metadata["tensors"]
    if not isinstance(descriptors, list) or len(descriptors) > MAX_ARRAYS:
        raise ValueError("invalid tensor list")
    for descriptor in descriptors:
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        if dtype not in DTYPES or not isinstance(shape, list) or len(shape) > 5:
            raise ValueError("invalid tensor descriptor")
        if any(type(value) is not int or value < 0 for value in shape):
            raise ValueError("invalid tensor shape")
        size = math.prod(shape) * np.dtype(dtype).itemsize
        if (
            descriptor["offset"] != offset
            or descriptor["size"] != size
            or offset + size > len(binary)
        ):
            raise ValueError("invalid tensor bounds")
        arrays.append(
            np.frombuffer(binary[offset : offset + size], dtype=dtype)
            .reshape(shape)
            .copy()
        )
        offset += size
    if offset != len(binary):
        raise ValueError("unexpected trailing tensor bytes")

    def decode(value):
        if isinstance(value, dict):
            if "__tensor__" in value:
                index = value["__tensor__"]
                if (
                    set(value) != {"__tensor__"}
                    or type(index) is not int
                    or not 0 <= index < len(arrays)
                ):
                    raise ValueError("invalid tensor reference")
                return arrays[index]
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value

    frame = decode(metadata["frame"])
    if not isinstance(frame, dict) or float(frame.get("time", math.nan)) != timestamp:
        raise ValueError("frame/header timestamp mismatch")
    return frame
