# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tensor coordinate transforms used by the edge root projection model."""

import torch


def affine_transform_pts_cuda(pts, transform):
    point_count = pts.shape[0]
    homogeneous = torch.cat(
        [pts, torch.ones(point_count, 1, device=pts.device)], dim=1
    )
    transformed = torch.mm(transform, torch.t(homogeneous))
    return torch.t(transformed[:2, :])
