# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Adopt only the assigned edge cameras into the region coordinate system."""

from copy import deepcopy
from dt_common.spatial.calibration import spatial_calibration


def merge_edge_calibration(base, update, edge_id, camera_ids):
    result = deepcopy(base)
    by_id = {str(c["camera_id"]).removeprefix("camera/"): c for c in result["cameras"]}
    updates = {}
    for camera in spatial_calibration(update)["cameras"]:
        prefix = f"{edge_id}/camera/"
        if not str(camera["camera_id"]).startswith(prefix):
            raise ValueError("calibration result contains a camera from another edge")
        number = str(camera["camera_id"])[len(prefix) :]
        camera["camera_id"] = f"camera/{number}"
        updates[number] = camera
    for camera_id in map(str, camera_ids):
        if camera_id not in updates or camera_id not in by_id:
            raise ValueError(
                f"calibration result does not cover assigned camera/{camera_id}"
            )
        by_id[camera_id] = updates[camera_id]
    # The worker output must use the same USD-world/OpenCV coordinate convention.
    if update["coordinate_convention"] != base["coordinate_convention"]:
        raise ValueError("calibration coordinate convention differs from the region")
    result["cameras"] = [
        by_id[str(c["camera_id"]).removeprefix("camera/")] for c in result["cameras"]
    ]
    return result

