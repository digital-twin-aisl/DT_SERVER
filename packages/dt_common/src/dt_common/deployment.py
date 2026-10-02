# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Compatibility exports for shared snapshots; region merge policy belongs to manager."""

from .spatial.calibration import spatial_calibration as spatial_calibration
from .infrastructure.deployment import (
    read_deployment_calibration as read_deployment_calibration,
    materialize_deployment as materialize_deployment,
)
