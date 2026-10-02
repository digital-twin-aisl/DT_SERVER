# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Spatial-only projection of a calibration result, without device policy or IO."""

from copy import deepcopy

CAMERA_FIELDS = (
    "camera_id",
    "camera_matrix",
    "distortion_coefficients",
    "undistorted_camera_matrix",
    "world_to_camera",
    "camera_to_world",
    "position_m",
    "source_image_size",
)


def spatial_calibration(document):
    return {
        "schema_version": 1,
        "coordinate_convention": deepcopy(document["coordinate_convention"]),
        "cameras": [
            {key: deepcopy(camera[key]) for key in CAMERA_FIELDS if key in camera}
            for camera in document["cameras"]
        ],
    }


