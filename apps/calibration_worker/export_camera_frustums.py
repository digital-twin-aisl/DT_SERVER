#!/usr/bin/env python3
"""Export calibration-result camera poses as USD wireframe frustums."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Sequence

import cv2
import numpy as np


CAMERA_COLORS = (
    (0.16, 0.62, 1.00),
    (1.00, 0.46, 0.22),
    (0.35, 0.80, 0.42),
    (0.75, 0.42, 1.00),
    (1.00, 0.78, 0.20),
    (0.12, 0.82, 0.78),
    (1.00, 0.38, 0.62),
    (0.62, 0.72, 0.18),
)


def _matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    return matrix


def _camera_image_size(camera: dict[str, Any], intrinsic: np.ndarray) -> tuple[int, int]:
    size = camera.get("source_image_size")
    if (
        isinstance(size, list)
        and len(size) == 2
        and all(float(value) > 0 for value in size)
    ):
        return int(size[0]), int(size[1])

    source_image = camera.get("source_image")
    if isinstance(source_image, str) and source_image:
        image = cv2.imread(source_image, cv2.IMREAD_UNCHANGED)
        if image is not None:
            height, width = image.shape[:2]
            return width, height

    # Older offline results did not persist source_image_size. Principal points
    # are normally near the image centre, so this remains a useful visual-only
    # fallback when the original capture is no longer available.
    width = max(1, int(round(2.0 * intrinsic[0, 2])))
    height = max(1, int(round(2.0 * intrinsic[1, 2])))
    return width, height


def camera_frustum_world_points(
    camera: dict[str, Any],
    depth_m: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return origin, four far-plane corners, and forward point in USD world."""
    if not math.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("frustum depth must be a positive finite value")
    camera_id = str(camera.get("camera_id") or "camera")
    intrinsic = _matrix(camera.get("camera_matrix"), (3, 3), f"{camera_id}.camera_matrix")
    camera_to_world = _matrix(
        camera.get("camera_to_world"),
        (4, 4),
        f"{camera_id}.camera_to_world",
    )
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError(f"{camera_id} focal lengths must be positive")
    width, height = _camera_image_size(camera, intrinsic)

    pixels = np.asarray(
        ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height)),
        dtype=np.float64,
    )
    far_corners = np.column_stack(
        (
            (pixels[:, 0] - cx) / fx * depth_m,
            (pixels[:, 1] - cy) / fy * depth_m,
            np.full(4, depth_m),
        )
    )
    camera_points = np.vstack(
        (
            np.zeros((1, 3), dtype=np.float64),
            far_corners,
            np.asarray(((0.0, 0.0, depth_m * 1.15),), dtype=np.float64),
        )
    )
    homogeneous = np.column_stack((camera_points, np.ones(len(camera_points))))
    world_points = (camera_to_world @ homogeneous.T).T[:, :3]
    return world_points, (width, height)


def _safe_prim_name(camera_id: str, index: int) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", camera_id).strip("_")
    if not token:
        token = f"Camera_{index}"
    if token[0].isdigit():
        token = f"Camera_{token}"
    return token


def _author_linear_curves(
    stage: Any,
    path: str,
    curves: Sequence[Sequence[np.ndarray]],
    color: tuple[float, float, float],
    width_m: float,
) -> None:
    from pxr import Gf, UsdGeom, Vt

    geometry = UsdGeom.BasisCurves.Define(stage, path)
    geometry.CreateTypeAttr(UsdGeom.Tokens.linear)
    geometry.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    geometry.CreateCurveVertexCountsAttr(
        Vt.IntArray([len(curve) for curve in curves])
    )
    points = [
        Gf.Vec3f(*(float(value) for value in point))
        for curve in curves
        for point in curve
    ]
    geometry.CreatePointsAttr(Vt.Vec3fArray(points))
    geometry.CreateWidthsAttr(Vt.FloatArray([float(width_m)]))
    geometry.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    geometry.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
        Vt.Vec3fArray([Gf.Vec3f(*color)])
    )


def _author_world_axes(stage: Any, root_path: str, length_m: float, width_m: float) -> None:
    origin = np.zeros(3, dtype=np.float64)
    axes = (
        ("X", np.asarray((length_m, 0.0, 0.0)), (1.0, 0.1, 0.1)),
        ("Y", np.asarray((0.0, length_m, 0.0)), (0.1, 1.0, 0.1)),
        ("Z", np.asarray((0.0, 0.0, length_m)), (0.1, 0.3, 1.0)),
    )
    for name, endpoint, color in axes:
        _author_linear_curves(
            stage,
            f"{root_path}/{name}",
            ((origin, endpoint),),
            color,
            width_m,
        )


def export_camera_frustums(
    result_path: str | Path,
    output_path: str | Path | None = None,
    *,
    frustum_depth_m: float = 2.0,
    line_width_m: float = 0.035,
    include_marker_tree: bool = True,
) -> Path:
    """Create a Z-up, metre-based USD stage from a calibration result JSON."""
    try:
        from pxr import Gf, Usd, UsdGeom, Vt
    except ImportError as exc:
        raise RuntimeError(
            "Pixar USD Python modules are required; use Isaac Sim python or install usd-core"
        ) from exc

    source = Path(result_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"calibration result does not exist: {source}")
    result = json.loads(source.read_text(encoding="utf-8"))
    cameras = result.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("calibration result contains no cameras")
    if not math.isfinite(line_width_m) or line_width_m <= 0:
        raise ValueError("line width must be a positive finite value")

    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_name(f"{source.stem}_cameras.usda")
    )
    if destination.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        destination = destination.with_suffix(".usda")
    destination.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_path = "/CalibrationVisualization"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)
    root.SetCustomDataByKey("calibration:sourceResult", str(source))
    root.SetCustomDataByKey("calibration:frustumDepthM", float(frustum_depth_m))

    marker_tree = result.get("marker_tree")
    if include_marker_tree and isinstance(marker_tree, str) and marker_tree:
        marker_path = Path(marker_tree).expanduser().resolve()
        if marker_path.is_file():
            marker_reference = UsdGeom.Xform.Define(
                stage, f"{root_path}/MarkerTree"
            ).GetPrim()
            relative_asset = os.path.relpath(marker_path, destination.parent)
            marker_reference.GetReferences().AddReference(relative_asset)

    cameras_root = UsdGeom.Xform.Define(stage, f"{root_path}/Cameras")
    used_names: set[str] = set()
    for index, camera in enumerate(cameras, 1):
        if not isinstance(camera, dict):
            raise ValueError(f"cameras[{index - 1}] must be an object")
        camera_id = str(camera.get("camera_id") or f"camera/{index}")
        base_name = _safe_prim_name(camera_id, index)
        prim_name = base_name
        suffix = 2
        while prim_name in used_names:
            prim_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(prim_name)

        points, image_size = camera_frustum_world_points(camera, frustum_depth_m)
        origin = points[0]
        corners = points[1:5]
        forward = points[5]
        camera_path = f"{cameras_root.GetPath()}/{prim_name}"
        camera_prim = UsdGeom.Xform.Define(stage, camera_path).GetPrim()
        camera_prim.SetCustomDataByKey("calibration:cameraId", camera_id)
        camera_prim.SetCustomDataByKey(
            "calibration:imageSize", Vt.IntArray(list(image_size))
        )
        camera_prim.SetCustomDataByKey(
            "calibration:positionM",
            Vt.DoubleArray([float(value) for value in origin]),
        )

        color = CAMERA_COLORS[(index - 1) % len(CAMERA_COLORS)]
        curves = [
            (origin, corners[0]),
            (origin, corners[1]),
            (origin, corners[2]),
            (origin, corners[3]),
            (corners[0], corners[1], corners[2], corners[3], corners[0]),
            (origin, forward),
        ]
        _author_linear_curves(
            stage,
            f"{camera_path}/Frustum",
            curves,
            color,
            line_width_m,
        )

        origin_marker = UsdGeom.Sphere.Define(stage, f"{camera_path}/Origin")
        origin_marker.CreateRadiusAttr(max(line_width_m * 2.25, 0.04))
        origin_marker.AddTranslateOp().Set(Gf.Vec3d(*origin))
        origin_marker.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
            Vt.Vec3fArray([Gf.Vec3f(*color)])
        )

    _author_world_axes(
        stage,
        f"{root_path}/WorldAxes",
        max(1.0, frustum_depth_m),
        line_width_m,
    )
    stage.GetRootLayer().Save()
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export calibration camera poses as USD wireframe frustums"
    )
    parser.add_argument("result", help="calibration_result_*.json")
    parser.add_argument("--output", help="output .usd/.usda/.usdc path")
    parser.add_argument("--frustum-depth-m", type=float, default=2.0)
    parser.add_argument("--line-width-m", type=float, default=0.035)
    parser.add_argument(
        "--no-marker-tree",
        action="store_true",
        help="do not reference the result's marker-tree USD",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        output = export_camera_frustums(
            args.result,
            args.output,
            frustum_depth_m=args.frustum_depth_m,
            line_width_m=args.line_width_m,
            include_marker_tree=not args.no_marker_tree,
        )
    except Exception as exc:
        print(f"USD export failed: {exc}")
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
