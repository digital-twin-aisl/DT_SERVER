"""ArUco marker image and USD board generation.

This module deliberately keeps Kit UI code out of the asset generation path so it
can also be called from Isaac Sim scripts and covered by ordinary Python tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any


DICTIONARY_NAMES = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
)


@dataclass(frozen=True)
class GeneratedBoard:
    usd_path: Path
    texture_path: Path
    dictionary_name: str
    marker_id: int
    marker_length_mm: float
    board_side_length_mm: float


@dataclass(frozen=True)
class GeneratedMarkerTree:
    tree_path: Path
    marker_asset_path: Path
    texture_path: Path
    marker_prim_path: str
    marker_count: int


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV with the aruco module is required. Install opencv-contrib-python "
            "into the Isaac Sim Python environment."
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "The installed OpenCV has no aruco module. Install "
            "opencv-contrib-python into the Isaac Sim Python environment."
        )
    return cv2


def _dictionary(cv2: Any, dictionary_name: str) -> Any:
    if dictionary_name not in DICTIONARY_NAMES:
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise RuntimeError(
            f"This OpenCV build does not provide {dictionary_name}."
        )
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _validate_marker_id(dictionary: Any, marker_id: int) -> None:
    marker_count = int(dictionary.bytesList.shape[0])
    if marker_id < 0 or marker_id >= marker_count:
        raise ValueError(
            f"Marker ID must be between 0 and {marker_count - 1} for this dictionary."
        )


def generate_marker_image(
    dictionary_name: str,
    marker_id: int,
    output_path: str | Path,
    pixels: int = 1024,
) -> Path:
    """Write a crisp ArUco marker texture without physical-size padding."""
    if pixels < 64:
        raise ValueError("Marker image resolution must be at least 64 pixels.")

    cv2 = _load_cv2()
    dictionary = _dictionary(cv2, dictionary_name)
    _validate_marker_id(dictionary, int(marker_id))

    if hasattr(cv2.aruco, "generateImageMarker"):
        image = cv2.aruco.generateImageMarker(
            dictionary, int(marker_id), int(pixels), borderBits=1
        )
    else:
        image = cv2.aruco.drawMarker(
            dictionary, int(marker_id), int(pixels), borderBits=1
        )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Failed to write marker texture: {output}")
    return output


def generate_board_usd(
    dictionary_name: str,
    marker_id: int,
    marker_length_mm: float,
    output_path: str | Path,
    *,
    thickness_mm: float = 1.0,
    texture_pixels: int = 1024,
    board_to_marker_ratio: float = 1.05,
    overwrite: bool = False,
) -> GeneratedBoard:
    """Create a Z-up, metre-based USD board and its adjacent marker texture.

    The marker plane is exactly ``marker_length_mm`` wide. The thin white backing
    is ``board_to_marker_ratio`` times wider and has a collision shape but no rigid
    body API, so it behaves as a static collider in an Isaac Sim stage.
    """
    if not math.isfinite(float(marker_length_mm)) or marker_length_mm <= 0:
        raise ValueError("Marker length must be greater than 0 mm.")
    if not math.isfinite(float(thickness_mm)) or thickness_mm <= 0:
        raise ValueError("Board thickness must be greater than 0 mm.")
    if not math.isfinite(float(board_to_marker_ratio)) or board_to_marker_ratio < 1.0:
        raise ValueError("Board-to-marker ratio must be at least 1.0.")

    if not str(output_path).strip():
        raise ValueError("Output USD path cannot be empty.")
    usd_path = Path(output_path).expanduser().resolve()
    if usd_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        usd_path = usd_path.with_suffix(".usd")
    safe_dictionary = re.sub(r"[^A-Za-z0-9_-]", "_", dictionary_name.lower())
    safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", usd_path.stem)
    texture_path = usd_path.parent / "textures" / (
        f"{safe_stem}_aruco_{safe_dictionary}_{int(marker_id)}.png"
    )
    existing = [path for path in (usd_path, texture_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {names}")

    # Validate the marker before creating either output directory or USD file.
    cv2 = _load_cv2()
    dictionary = _dictionary(cv2, dictionary_name)
    _validate_marker_id(dictionary, int(marker_id))

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:
        raise RuntimeError(
            "Pixar USD Python modules are required; run this generator with Isaac "
            "Sim's Python interpreter."
        ) from exc

    usd_path.parent.mkdir(parents=True, exist_ok=True)
    generate_marker_image(
        dictionary_name,
        int(marker_id),
        texture_path,
        pixels=texture_pixels,
    )

    marker_length_mm = float(marker_length_mm)
    board_side_length_mm = marker_length_mm * float(board_to_marker_ratio)
    marker_side_m = marker_length_mm / 1000.0
    board_side_m = board_side_length_mm / 1000.0
    thickness_m = float(thickness_mm) / 1000.0
    marker_half = marker_side_m / 2.0
    top_z = thickness_m / 2.0 + min(1.0e-5, thickness_m * 0.01)

    stage = Usd.Stage.CreateNew(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not create USD stage: {usd_path}")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/ArUcoBoard")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("aruco:dictionary", dictionary_name)
    root.GetPrim().SetCustomDataByKey("aruco:markerId", int(marker_id))
    root.GetPrim().SetCustomDataByKey("aruco:markerLengthMm", marker_length_mm)
    root.GetPrim().SetCustomDataByKey(
        "aruco:boardSideLengthMm", board_side_length_mm
    )

    backing = UsdGeom.Cube.Define(stage, "/ArUcoBoard/Backing")
    backing.CreateSizeAttr(1.0)
    backing.AddScaleOp().Set(Gf.Vec3d(board_side_m, board_side_m, thickness_m))
    backing.CreateDisplayColorAttr([Gf.Vec3f(1.0, 1.0, 1.0)])
    UsdPhysics.CollisionAPI.Apply(backing.GetPrim())

    marker = UsdGeom.Mesh.Define(stage, "/ArUcoBoard/Marker")
    marker.CreatePointsAttr(
        [
            Gf.Vec3f(-marker_half, -marker_half, top_z),
            Gf.Vec3f(marker_half, -marker_half, top_z),
            Gf.Vec3f(marker_half, marker_half, top_z),
            Gf.Vec3f(-marker_half, marker_half, top_z),
        ]
    )
    marker.CreateFaceVertexCountsAttr([4])
    marker.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    marker.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    marker.CreateExtentAttr(
        [
            Gf.Vec3f(-marker_half, -marker_half, top_z),
            Gf.Vec3f(marker_half, marker_half, top_z),
        ]
    )
    st = UsdGeom.PrimvarsAPI(marker).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    # PNG rows start at the top, so V is flipped to preserve marker orientation.
    st.Set(
        [
            Gf.Vec2f(0.0, 1.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(0.0, 0.0),
        ]
    )

    material = UsdShade.Material.Define(stage, "/ArUcoBoard/MarkerMaterial")
    surface = UsdShade.Shader.Define(
        stage, "/ArUcoBoard/MarkerMaterial/PreviewSurface"
    )
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    texture = UsdShade.Shader.Define(
        stage, "/ArUcoBoard/MarkerMaterial/MarkerTexture"
    )
    texture.CreateIdAttr("UsdUVTexture")
    relative_texture = texture_path.relative_to(usd_path.parent).as_posix()
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(relative_texture)
    )
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    reader = UsdShade.Shader.Define(
        stage, "/ArUcoBoard/MarkerMaterial/PrimvarReader"
    )
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(
        surface.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()
    return GeneratedBoard(
        usd_path=usd_path,
        texture_path=texture_path,
        dictionary_name=dictionary_name,
        marker_id=int(marker_id),
        marker_length_mm=marker_length_mm,
        board_side_length_mm=board_side_length_mm,
    )


def append_marker_to_tree_usd(
    dictionary_name: str,
    marker_id: int,
    marker_length_mm: float,
    tree_path: str | Path,
    *,
    thickness_mm: float = 1.0,
    texture_pixels: int = 1024,
    board_to_marker_ratio: float = 1.05,
    gap_mm: float = 10.0,
    position_m: tuple[float, float, float] | None = None,
    orientation_wxyz: tuple[float, float, float, float] | None = None,
) -> GeneratedMarkerTree:
    """Append one marker board to a reusable USD collection tree.

    The tree is the public asset users load into a Stage. Individual board USDs
    and textures live below its ``assets`` directory as implementation details.
    When a pose is supplied, it is authored directly on the marker child and is
    therefore restored whenever the tree is loaded. Without a pose, children are
    laid out from left to right with ``gap_mm`` clearance.
    """
    if not str(tree_path).strip():
        raise ValueError("Marker tree USD path cannot be empty.")
    if not math.isfinite(float(gap_mm)) or gap_mm < 0:
        raise ValueError("Marker gap cannot be negative.")
    if position_m is not None:
        if len(position_m) != 3 or not all(
            math.isfinite(float(value)) for value in position_m
        ):
            raise ValueError("Marker position must contain three finite values.")
    if orientation_wxyz is not None:
        if len(orientation_wxyz) != 4 or not all(
            math.isfinite(float(value)) for value in orientation_wxyz
        ):
            raise ValueError("Marker orientation must contain four finite values.")
        quaternion_length = math.sqrt(
            sum(float(value) ** 2 for value in orientation_wxyz)
        )
        if quaternion_length <= 1.0e-12:
            raise ValueError("Marker orientation quaternion cannot be zero.")

    tree_path = Path(tree_path).expanduser().resolve()
    if tree_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        tree_path = tree_path.with_suffix(".usd")
    tree_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError(
            "Pixar USD Python modules are required; run this generator with Isaac "
            "Sim's Python interpreter."
        ) from exc

    if tree_path.exists():
        tree_stage = Usd.Stage.Open(str(tree_path))
        if tree_stage is None:
            raise RuntimeError(f"Could not open marker tree: {tree_path}")
        root = tree_stage.GetDefaultPrim()
        if not root.IsValid() or root.GetPath().pathString != "/ArUcoMarkerTree":
            raise ValueError(
                f"Existing USD is not an ArUco marker tree: {tree_path}"
            )
    else:
        tree_stage = Usd.Stage.CreateNew(str(tree_path))
        if tree_stage is None:
            raise RuntimeError(f"Could not create marker tree: {tree_path}")
        UsdGeom.SetStageMetersPerUnit(tree_stage, 1.0)
        UsdGeom.SetStageUpAxis(tree_stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(tree_stage, "/ArUcoMarkerTree").GetPrim()
        tree_stage.SetDefaultPrim(root)
        root.SetCustomDataByKey("aruco:assetType", "markerTree")
        UsdGeom.Xform.Define(tree_stage, "/ArUcoMarkerTree/Markers")

    markers = tree_stage.GetPrimAtPath("/ArUcoMarkerTree/Markers")
    if not markers.IsValid():
        markers = UsdGeom.Xform.Define(
            tree_stage, "/ArUcoMarkerTree/Markers"
        ).GetPrim()
    children = list(markers.GetChildren())

    safe_dictionary = re.sub(
        r"[^A-Za-z0-9_]", "_", dictionary_name.removeprefix("DICT_")
    )
    base_name = f"Marker_{safe_dictionary}_{int(marker_id)}"
    marker_name = base_name
    suffix = 2
    assets_dir = tree_path.parent / "assets"
    while (
        tree_stage.GetPrimAtPath(f"/ArUcoMarkerTree/Markers/{marker_name}").IsValid()
        or (assets_dir / f"{marker_name}.usd").exists()
    ):
        marker_name = f"{base_name}_{suffix}"
        suffix += 1

    marker_asset_path = assets_dir / f"{marker_name}.usd"
    board = generate_board_usd(
        dictionary_name=dictionary_name,
        marker_id=marker_id,
        marker_length_mm=marker_length_mm,
        output_path=marker_asset_path,
        thickness_mm=thickness_mm,
        texture_pixels=texture_pixels,
        board_to_marker_ratio=board_to_marker_ratio,
    )

    new_board_width_mm = board.board_side_length_mm
    if position_m is not None:
        center_x_mm = float(position_m[0]) * 1000.0
    elif children:
        right_edges = []
        for child in children:
            center_x_mm = float(
                child.GetCustomDataByKey("aruco:centerXmm") or 0.0
            )
            width_mm = float(
                child.GetCustomDataByKey("aruco:boardSideLengthMm") or 0.0
            )
            right_edges.append(center_x_mm + width_mm / 2.0)
        center_x_mm = max(right_edges) + float(gap_mm) + new_board_width_mm / 2.0
    else:
        center_x_mm = 0.0

    marker_prim_path = f"/ArUcoMarkerTree/Markers/{marker_name}"
    marker_xform = UsdGeom.Xform.Define(tree_stage, marker_prim_path)
    relative_asset = marker_asset_path.relative_to(tree_path.parent).as_posix()
    marker_xform.GetPrim().GetReferences().AddReference(relative_asset)
    if position_m is None:
        marker_position = Gf.Vec3d(center_x_mm / 1000.0, 0.0, 0.0)
    else:
        marker_position = Gf.Vec3d(*(float(value) for value in position_m))
    marker_xform.AddTranslateOp().Set(marker_position)
    if orientation_wxyz is not None:
        real, x, y, z = (float(value) for value in orientation_wxyz)
        marker_orientation = Gf.Quatd(real, Gf.Vec3d(x, y, z)).GetNormalized()
        marker_xform.AddOrientOp(
            precision=UsdGeom.XformOp.PrecisionDouble
        ).Set(marker_orientation)
    marker_xform.GetPrim().SetCustomDataByKey("aruco:index", len(children))
    marker_xform.GetPrim().SetCustomDataByKey("aruco:dictionary", dictionary_name)
    marker_xform.GetPrim().SetCustomDataByKey("aruco:markerId", int(marker_id))
    marker_xform.GetPrim().SetCustomDataByKey(
        "aruco:markerLengthMm", float(marker_length_mm)
    )
    marker_xform.GetPrim().SetCustomDataByKey(
        "aruco:boardSideLengthMm", new_board_width_mm
    )
    marker_xform.GetPrim().SetCustomDataByKey("aruco:centerXmm", center_x_mm)

    marker_count = len(children) + 1
    root.SetCustomDataByKey("aruco:markerCount", marker_count)
    tree_stage.GetRootLayer().Save()
    return GeneratedMarkerTree(
        tree_path=tree_path,
        marker_asset_path=board.usd_path,
        texture_path=board.texture_path,
        marker_prim_path=marker_prim_path,
        marker_count=marker_count,
    )
