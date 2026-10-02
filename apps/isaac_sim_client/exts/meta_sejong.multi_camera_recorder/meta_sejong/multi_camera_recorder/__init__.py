# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Isaac Sim multi-camera recorder extension package."""

try:
    from .extension import MultiCameraRecorderExtension
except ModuleNotFoundError as exc:
    # Keep the hard-coded calibration helpers testable with normal Python.
    if exc.name != "omni":
        raise
    MultiCameraRecorderExtension = None

__all__ = ["MultiCameraRecorderExtension"]
