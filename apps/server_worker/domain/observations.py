# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Server-local root association value object."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RootCandidate:
    """한 Edge 안에서 Global ID와 연결된 root 후보."""

    global_id: int
    edge_id: str
    edge_index: int
    root_index: int
    confidence: float
    reprojection_error: float
    observation_count: int

