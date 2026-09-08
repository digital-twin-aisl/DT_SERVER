"""Fingerprint the camera parameters actually used by projection on each host."""

import hashlib
import json

import numpy as np


def calibration_digest(cameras):
    records = []
    for camera in cameras:
        record = {"id": int(camera["id"])}
        for key in ("R", "T", "fx", "fy", "cx", "cy", "k", "p"):
            value = np.asarray(camera[key], dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError("non-finite camera calibration")
            # Canonical decimal values avoid platform byte order and -0.0.
            record[key] = [
                float(format(float(v), ".10g")) if v != 0 else 0.0
                for v in value.reshape(-1)
            ]
        records.append(record)
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
