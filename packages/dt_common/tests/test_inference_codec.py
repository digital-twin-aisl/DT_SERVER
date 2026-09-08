import struct

import numpy as np
import pytest

from dt_common.inference_codec import HEADER, MAX_RAW_BYTES, decode_frame, encode_frame


def test_roundtrip_preserves_nested_arrays_and_types():
    frame = {
        "time": 1.25,
        "camera_ids": [2, 4],
        "allheatmaps": [np.arange(24, dtype=np.uint8).reshape(1, 2, 3, 4)],
        "roots": np.zeros((1, 10, 5), dtype=np.float32),
        "reid": [
            {"feature": np.array([0.1, 0.5], dtype=np.float32), "bbox": [1, 2, 3, 4]}
        ],
    }
    wire = encode_frame(frame)
    restored = decode_frame(wire)
    assert wire[:4] == b"ZNH2"
    assert restored["time"] == 1.25
    np.testing.assert_array_equal(restored["allheatmaps"][0], frame["allheatmaps"][0])
    np.testing.assert_array_equal(
        restored["reid"][0]["feature"], frame["reid"][0]["feature"]
    )
    assert restored["roots"].dtype == np.float32


@pytest.mark.parametrize(
    "wire",
    [
        b"",
        b"ZNH1" + b"x" * 50,
        HEADER.pack(b"ZNH2", 1.0, MAX_RAW_BYTES + 1) + b"x",
        HEADER.pack(b"ZNH2", float("nan"), 5) + b"x",
    ],
)
def test_rejects_legacy_and_invalid_headers(wire):
    with pytest.raises(ValueError):
        decode_frame(wire)


def test_rejects_object_arrays_and_reserved_keys():
    for value in (np.array([object()], dtype=object), {"__tensor__": 0}):
        with pytest.raises(ValueError):
            encode_frame({"time": 1.0, "value": value})


def test_rejects_header_and_body_timestamp_mismatch():
    wire = bytearray(encode_frame({"time": 1.0}))
    wire[4:12] = struct.pack("!d", 2.0)
    with pytest.raises(ValueError, match="timestamp mismatch"):
        decode_frame(bytes(wire))


def test_rejects_false_decompressed_size():
    wire = encode_frame({"time": 1.0})
    magic, stamp, size = HEADER.unpack_from(wire)
    with pytest.raises(ValueError, match="size"):
        decode_frame(HEADER.pack(magic, stamp, size + 1) + wire[HEADER.size :])
