from pathlib import Path

import pytest
import cv2
from pxr import Usd, UsdGeom

from meta_sejong.aruco_board.generator import (
    append_marker_to_tree_usd,
    generate_board_usd,
)


def test_generates_metric_board_and_texture(tmp_path: Path) -> None:
    result = generate_board_usd(
        "DICT_4X4_50", 7, 80.0, tmp_path / "board.usda"
    )

    assert result.usd_path.exists()
    assert result.texture_path.exists()
    stage = Usd.Stage.Open(str(result.usd_path))
    assert stage.GetDefaultPrim().GetPath().pathString == "/ArUcoBoard"
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)
    assert stage.GetPrimAtPath("/ArUcoBoard/Marker").IsValid()
    assert stage.GetPrimAtPath("/ArUcoBoard/Backing").IsValid()
    backing = UsdGeom.Cube.Get(stage, "/ArUcoBoard/Backing")
    scale = backing.GetOrderedXformOps()[0].Get()
    assert scale[0] == pytest.approx(0.084)
    assert scale[1] == pytest.approx(0.084)
    assert result.marker_length_mm == pytest.approx(80.0)
    assert result.board_side_length_mm == pytest.approx(84.0)
    marker = UsdGeom.Mesh.Get(stage, "/ArUcoBoard/Marker")
    points = marker.GetPointsAttr().Get()
    marker_width = max(point[0] for point in points) - min(
        point[0] for point in points
    )
    assert marker_width == pytest.approx(0.08)

    image = cv2.imread(str(result.texture_path), cv2.IMREAD_GRAYSCALE)
    # The physical white detection margin is USD backing geometry, so emulate it
    # when validating the standalone marker texture.
    image = cv2.copyMakeBorder(
        image, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255
    )
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    )
    _, ids, _ = detector.detectMarkers(image)
    assert ids.ravel().tolist() == [7]


def test_rejects_id_outside_dictionary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 0 and 49"):
        generate_board_usd("DICT_4X4_50", 50, 80.0, tmp_path / "board.usd")


def test_does_not_overwrite_by_default(tmp_path: Path) -> None:
    output = tmp_path / "board.usd"
    generate_board_usd("DICT_6X6_250", 2, 100.0, output)
    with pytest.raises(FileExistsError):
        generate_board_usd("DICT_6X6_250", 2, 100.0, output)


def test_appends_multiple_markers_to_one_tree(tmp_path: Path) -> None:
    tree_path = tmp_path / "my_marker_tree.usda"
    first = append_marker_to_tree_usd(
        "DICT_4X4_50", 3, 100.0, tree_path
    )
    second = append_marker_to_tree_usd(
        "DICT_6X6_250", 12, 80.0, tree_path
    )

    assert first.marker_count == 1
    assert second.marker_count == 2
    assert first.marker_asset_path.exists()
    assert second.marker_asset_path.exists()

    stage = Usd.Stage.Open(str(tree_path))
    root = stage.GetDefaultPrim()
    assert root.GetPath().pathString == "/ArUcoMarkerTree"
    assert root.GetCustomDataByKey("aruco:markerCount") == 2
    children = list(stage.GetPrimAtPath("/ArUcoMarkerTree/Markers").GetChildren())
    assert len(children) == 2
    first_right_edge = 100.0 * 1.05 / 2.0
    second_left_edge = (
        children[1].GetCustomDataByKey("aruco:centerXmm")
        - 80.0 * 1.05 / 2.0
    )
    assert second_left_edge - first_right_edge == pytest.approx(10.0)


def test_persists_authored_marker_pose_in_tree(tmp_path: Path) -> None:
    tree_path = tmp_path / "positioned_tree.usda"
    result = append_marker_to_tree_usd(
        "DICT_7X7_100",
        9,
        120.0,
        tree_path,
        position_m=(1.25, -0.5, 2.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    stage = Usd.Stage.Open(str(tree_path))
    marker = UsdGeom.Xform.Get(stage, result.marker_prim_path)
    ops = marker.GetOrderedXformOps()
    assert ops[0].Get() == pytest.approx((1.25, -0.5, 2.0))
    assert ops[1].Get().GetReal() == pytest.approx(1.0)
