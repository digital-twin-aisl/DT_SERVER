# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Portable run snapshots. Each host links its own assets; source files are immutable."""

import json
from pathlib import Path
from dt_common.spatial.calibration import spatial_calibration


def read_deployment_calibration(deployment):
    path = Path(deployment).resolve()
    manifest = json.loads(path.read_text())
    source = (path.parent / manifest["calibration_result"]).resolve()
    return spatial_calibration(json.loads(source.read_text()))



def materialize_deployment(deployment, destination, calibration):
    source = Path(deployment).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(source.read_text())
    for key in ("ground_usd", "ground_cache"):
        if manifest.get("scene", {}).get(key):
            original = (source.parent / manifest["scene"][key]).resolve()
            if not original.is_file():
                raise ValueError(f"missing spatial asset: {original}")
            filename = "ground.usd" if key == "ground_usd" else "ground.npz"
            link = destination / filename
            if not link.exists():
                link.symlink_to(original)
            elif link.resolve() != original:
                raise ValueError("spatial snapshot already refers to a different asset")
            manifest["scene"][key] = filename
    calibration_path = destination / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            spatial_calibration(calibration),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )
    manifest["calibration_result"] = "calibration.json"
    target = destination / "deployment.json"
    target.write_text(
        json.dumps(
            manifest, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2
        )
        + "\n"
    )
    return target

