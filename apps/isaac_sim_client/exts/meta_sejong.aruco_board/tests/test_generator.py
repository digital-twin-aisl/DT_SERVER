from pathlib import Path

import pytest
import cv2
from pxr import Gf, Usd, UsdGeom

from meta_sejong.aruco_board.generator import (
    append_marker_to_tree_usd,
    delete_marker_from_tree_usd,
    generate_board_usd,
    read_marker_tree_usd,
    save_marker_transform_to_tree_usd,
    update_marker_in_tree_usd,
)


def test_generates_metric_board_and_texture(tmp_path: Path) -> None:
    result = generate_board_usd("DICT_4X4_50", 7, 80.0, tmp_path / "board.usda")

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
    marker_width = max(point[0] for point in points) - min(point[0] for point in points)
    assert marker_width == pytest.approx(0.08)

    image = cv2.imread(str(result.texture_path), cv2.IMREAD_GRAYSCALE)
    # The physical white detection margin is USD backing geometry, so emulate it
    # when validating the standalone marker texture.
    image = cv2.copyMakeBorder(image, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)
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
    first = append_marker_to_tree_usd("DICT_4X4_50", 3, 100.0, tree_path)
    second = append_marker_to_tree_usd("DICT_6X6_250", 12, 80.0, tree_path)

    assert first.marker_count == 1
    assert second.marker_count == 2
    assert first.texture_path.exists()
    assert second.texture_path.exists()

    stage = Usd.Stage.Open(str(tree_path))
    root = stage.GetDefaultPrim()
    assert root.GetPath().pathString == "/ArUcoMarkerTree"
    assert root.GetCustomDataByKey("aruco:markerCount") == 2
    children = list(stage.GetPrimAtPath("/ArUcoMarkerTree/Markers").GetChildren())
    assert len(children) == 2
    assert stage.GetPrimAtPath(f"{children[0].GetPath()}/Backing").IsValid()
    assert stage.GetPrimAtPath(f"{children[0].GetPath()}/Marker").IsValid()
    assert not children[0].HasAuthoredReferences()
    assert not (tmp_path / "assets").exists()
    first_right_edge = 100.0 * 1.05 / 2.0
    second_left_edge = (
        children[1].GetCustomDataByKey("aruco:centerXmm") - 80.0 * 1.05 / 2.0
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


def test_rejects_duplicate_detection_identity(tmp_path: Path) -> None:
    tree_path = tmp_path / "unique_gcp_tree.usda"
    append_marker_to_tree_usd("DICT_4X4_50", 7, 100.0, tree_path)

    with pytest.raises(ValueError, match="already present"):
        append_marker_to_tree_usd("DICT_4X4_50", 7, 100.0, tree_path)


def test_recovers_empty_tree_file(tmp_path: Path) -> None:
    tree_path = tmp_path / "interrupted_tree.usda"
    empty_stage = Usd.Stage.CreateNew(str(tree_path))
    empty_stage.GetRootLayer().Save()
    del empty_stage

    result = append_marker_to_tree_usd("DICT_5X5_100", 8, 90.0, tree_path)

    stage = Usd.Stage.Open(str(tree_path))
    assert stage.GetDefaultPrim().GetPath().pathString == "/ArUcoMarkerTree"
    assert stage.GetPrimAtPath(result.marker_prim_path).IsValid()


def test_saves_session_marker_pose_only_to_tree(tmp_path: Path) -> None:
    tree_path = tmp_path / "gcp_tree.usda"
    marker = append_marker_to_tree_usd("DICT_6X6_250", 18, 110.0, tree_path)
    map_path = tmp_path / "campus.usda"
    stage = Usd.Stage.CreateNew(str(map_path))
    UsdGeom.Xform.Define(stage, "/World")
    stage.GetRootLayer().Save()
    campus_contents_before = stage.GetRootLayer().ExportToString()

    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        tree_reference = UsdGeom.Xform.Define(
            stage, "/World/ArUcoMarkerTreeEditor/Test/Tree"
        )
        tree_reference.GetPrim().GetReferences().AddReference(str(tree_path))
    marker_path = (
        "/World/ArUcoMarkerTreeEditor/Test/Tree"
        + marker.marker_prim_path.removeprefix("/ArUcoMarkerTree")
    )

    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        session_marker = UsdGeom.Xformable(stage.GetPrimAtPath(marker_path))
        session_pose = Gf.Matrix4d(1.0)
        session_pose.SetTranslate(Gf.Vec3d(3.0, -2.0, 1.25))
        session_marker.MakeMatrixXform().Set(session_pose)

    composed_pose = UsdGeom.Xformable(
        stage.GetPrimAtPath(marker_path)
    ).GetLocalTransformation()
    save_marker_transform_to_tree_usd(
        tree_path,
        marker.marker_prim_path,
        composed_pose,
    )

    assert stage.GetRootLayer().ExportToString() == campus_contents_before
    reopened_tree = Usd.Stage.Open(str(tree_path))
    saved_marker = UsdGeom.Xformable(
        reopened_tree.GetPrimAtPath(marker.marker_prim_path)
    )
    saved_position = saved_marker.GetLocalTransformation().ExtractTranslation()
    assert saved_position == pytest.approx((3.0, -2.0, 1.25))


def test_updates_marker_properties_inside_tree(tmp_path: Path) -> None:
    tree_path = tmp_path / "editable_tree.usda"
    marker = append_marker_to_tree_usd("DICT_4X4_50", 4, 100.0, tree_path)
    old_texture_path = marker.texture_path

    result = update_marker_in_tree_usd(
        tree_path,
        marker.marker_prim_path,
        "DICT_5X5_100",
        9,
        140.0,
    )

    stage = Usd.Stage.Open(str(tree_path))
    updated = stage.GetPrimAtPath(result.marker_prim_path)
    assert result.marker_prim_path.endswith("/Marker_5X5_100_9")
    assert updated.GetCustomDataByKey("aruco:dictionary") == "DICT_5X5_100"
    assert updated.GetCustomDataByKey("aruco:markerId") == 9
    assert updated.GetCustomDataByKey("aruco:markerLengthMm") == pytest.approx(140.0)
    assert stage.GetPrimAtPath(f"{result.marker_prim_path}/Backing").IsValid()
    assert result.texture_path.exists()
    assert not old_texture_path.exists()


def test_deletes_marker_and_texture_from_tree(tmp_path: Path) -> None:
    tree_path = tmp_path / "deletable_tree.usda"
    first = append_marker_to_tree_usd("DICT_4X4_50", 1, 100.0, tree_path)
    second = append_marker_to_tree_usd("DICT_4X4_50", 2, 100.0, tree_path)

    assert delete_marker_from_tree_usd(tree_path, first.marker_prim_path) == 1

    stage = Usd.Stage.Open(str(tree_path))
    assert not stage.GetPrimAtPath(first.marker_prim_path).IsValid()
    assert stage.GetPrimAtPath(second.marker_prim_path).IsValid()
    assert stage.GetDefaultPrim().GetCustomDataByKey("aruco:markerCount") == 1
    assert not first.texture_path.exists()

    entries = read_marker_tree_usd(tree_path)
    assert len(entries) == 1
    assert entries[0].prim_path == second.marker_prim_path
    assert entries[0].dictionary_name == "DICT_4X4_50"
    assert entries[0].marker_id == 2
    assert entries[0].marker_length_mm == pytest.approx(100.0)
