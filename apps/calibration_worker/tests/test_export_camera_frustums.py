import json

import numpy as np
from pxr import Usd, UsdGeom

from apps.calibration_worker.export_camera_frustums import (
    camera_frustum_world_points,
    export_camera_frustums,
)


def _camera(camera_to_world=None):
    return {
        "camera_id": "camera/1",
        "camera_matrix": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        "source_image_size": [100, 100],
        "camera_to_world": (
            np.eye(4).tolist() if camera_to_world is None else camera_to_world
        ),
    }


def test_camera_frustum_uses_opencv_forward_axis():
    points, image_size = camera_frustum_world_points(_camera(), 2.0)
    assert image_size == (100, 100)
    np.testing.assert_allclose(points[0], [0, 0, 0])
    np.testing.assert_allclose(points[1], [-1, -1, 2])
    np.testing.assert_allclose(points[3], [1, 1, 2])
    np.testing.assert_allclose(points[5], [0, 0, 2.3])


def test_export_camera_frustums_authors_stage_and_marker_reference(tmp_path):
    marker_path = tmp_path / "marker_tree.usda"
    marker_stage = Usd.Stage.CreateNew(str(marker_path))
    marker_root = UsdGeom.Xform.Define(marker_stage, "/ArUcoMarkerTree").GetPrim()
    marker_stage.SetDefaultPrim(marker_root)
    marker_stage.GetRootLayer().Save()

    result_path = tmp_path / "calibration_result.json"
    result_path.write_text(
        json.dumps(
            {
                "marker_tree": str(marker_path),
                "cameras": [_camera()],
            }
        ),
        encoding="utf-8",
    )
    output = export_camera_frustums(
        result_path,
        tmp_path / "cameras.usda",
        frustum_depth_m=2.0,
    )
    stage = Usd.Stage.Open(str(output))
    assert stage.GetDefaultPrim().GetPath().pathString == "/CalibrationVisualization"
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetPrimAtPath(
        "/CalibrationVisualization/MarkerTree"
    ).IsValid()
    frustum = UsdGeom.BasisCurves.Get(
        stage,
        "/CalibrationVisualization/Cameras/camera_1/Frustum",
    )
    assert frustum.GetPrim().IsValid()
    assert len(frustum.GetPointsAttr().Get()) == 15
