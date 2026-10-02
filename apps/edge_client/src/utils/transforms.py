# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Image transforms shared by edge preprocessing and calibration."""

import cv2
import numpy as np
import torch


def get_affine_transform(
    center,
    scale,
    rotation,
    output_size,
    shift=np.array([0, 0], dtype=np.float32),
    inverse=0,
    **legacy_options,
):
    if "inv" in legacy_options:
        inverse = legacy_options.pop("inv")
    if legacy_options:
        raise TypeError(f"Unsupported options: {', '.join(legacy_options)}")
    if isinstance(scale, torch.Tensor):
        scale = np.array(scale.cpu())
    if isinstance(center, torch.Tensor):
        center = np.array(center.cpu())
    if isinstance(rotation, torch.Tensor):
        rotation = np.array(rotation.cpu())
    if not isinstance(scale, (np.ndarray, list)):
        scale = np.array([scale, scale])

    scaled = scale * 200.0
    source_width, source_height = scaled
    output_width, output_height = output_size
    rotation_radians = np.pi * rotation / 180
    if source_width >= source_height:
        source_direction = _rotate_direction(
            [0, source_width * -0.5], rotation_radians
        )
        output_direction = np.array([0, output_width * -0.5], np.float32)
    else:
        source_direction = _rotate_direction(
            [source_height * -0.5, 0], rotation_radians
        )
        output_direction = np.array([output_height * -0.5, 0], np.float32)

    source = np.zeros((3, 2), dtype=np.float32)
    output = np.zeros((3, 2), dtype=np.float32)
    source[0] = center + scaled * shift
    source[1] = center + source_direction + scaled * shift
    output[0] = [output_width * 0.5, output_height * 0.5]
    output[1] = output[0] + output_direction
    source[2] = _third_point(source[0], source[1])
    output[2] = _third_point(output[0], output[1])
    if inverse:
        return cv2.getAffineTransform(output, source)
    return cv2.getAffineTransform(source, output)


def _third_point(first, second):
    direction = first - second
    return np.array(second) + np.array(
        [-direction[1], direction[0]], dtype=np.float32
    )


def _rotate_direction(point, radians):
    sine, cosine = np.sin(radians), np.cos(radians)
    return [
        point[0] * cosine - point[1] * sine,
        point[0] * sine + point[1] * cosine,
    ]


def get_scale(image_size, resized_size):
    width, height = image_size
    resized_width, resized_height = resized_size
    if width / resized_width < height / resized_height:
        padded_width = height / resized_height * resized_width
        padded_height = height
    else:
        padded_width = width
        padded_height = width / resized_width * resized_height
    return np.array(
        [padded_width / 200.0, padded_height / 200.0], dtype=np.float32
    )
