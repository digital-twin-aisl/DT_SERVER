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


DICTIONARY_LABELS = ("4x4", "5x5", "6x6", "7x7")
DICTIONARY_NAMES = tuple(
    f"DICT_{label.upper()}_250" for label in DICTIONARY_LABELS
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
    texture_path: Path
    marker_prim_path: str
    marker_count: int


@dataclass(frozen=True)
class MarkerTreeEntry:
    prim_path: str
    dictionary_name: str
    marker_id: int
    marker_length_mm: float
    local_transform: Any


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
        raise RuntimeError(f"This OpenCV build does not provide {dictionary_name}.")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _validate_marker_id(dictionary: Any, marker_id: int) -> None:
    marker_count = int(dictionary.bytesList.shape[0])
    if marker_id < 0 or marker_id >= marker_count:
        raise ValueError(
            f"Marker ID must be between 0 and {marker_count - 1} for this dictionary."
        )


def _marker_dictionary_token(dictionary_name: str) -> str:
    """Return the marker-size portion used in generated tree Prim names."""
    dictionary_size = dictionary_name.removeprefix("DICT_").removesuffix("_250")
    return re.sub(r"[^A-Za-z0-9_]", "_", dictionary_size)


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


def _author_board_in_stage(
    stage: Any,
    root_path: str,
    dictionary_name: str,
    marker_id: int,
    marker_length_mm: float,
    thickness_mm: float,
    board_to_marker_ratio: float,
    texture_asset_path: str,
) -> Any:
    """Author one complete board below ``root_path`` in an existing stage.

    ``texture_asset_path`` must be relative to the USD layer containing the
    stage.  Keeping the geometry, material, and metadata below the marker root
    lets a marker tree be edited or deleted as one normal USD subtree.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    board_side_length_mm = marker_length_mm * board_to_marker_ratio
    marker_side_m = marker_length_mm / 1000.0
    board_side_m = board_side_length_mm / 1000.0
    thickness_m = thickness_mm / 1000.0
    marker_half = marker_side_m / 2.0
    top_z = thickness_m / 2.0 + min(1.0e-5, thickness_m * 0.01)

    root = UsdGeom.Xform.Define(stage, root_path)
    root_prim = root.GetPrim()
    root_prim.SetCustomDataByKey("aruco:dictionary", dictionary_name)
    root_prim.SetCustomDataByKey("aruco:markerId", int(marker_id))
    root_prim.SetCustomDataByKey("aruco:markerLengthMm", marker_length_mm)
    root_prim.SetCustomDataByKey("aruco:boardSideLengthMm", board_side_length_mm)

    backing = UsdGeom.Cube.Define(stage, f"{root_path}/Backing")
    backing.CreateSizeAttr(1.0)
    backing.AddScaleOp().Set(Gf.Vec3d(board_side_m, board_side_m, thickness_m))
    backing.CreateDisplayColorAttr([Gf.Vec3f(1.0, 1.0, 1.0)])
    UsdPhysics.CollisionAPI.Apply(backing.GetPrim())

    marker = UsdGeom.Mesh.Define(stage, f"{root_path}/Marker")
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

    material_path = f"{root_path}/MarkerMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    texture = UsdShade.Shader.Define(stage, f"{material_path}/MarkerTexture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(texture_asset_path)
    )
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    reader = UsdShade.Shader.Define(stage, f"{material_path}/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(material)
    return root


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
    texture_path = (
        usd_path.parent
        / "textures"
        / (f"{safe_stem}_aruco_{safe_dictionary}_{int(marker_id)}.png")
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
        from pxr import Usd, UsdGeom
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
    stage = Usd.Stage.CreateNew(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not create USD stage: {usd_path}")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    relative_texture = texture_path.relative_to(usd_path.parent).as_posix()
    root = _author_board_in_stage(
        stage,
        "/ArUcoBoard",
        dictionary_name,
        int(marker_id),
        marker_length_mm,
        float(thickness_mm),
        float(board_to_marker_ratio),
        relative_texture,
    )
    stage.SetDefaultPrim(root.GetPrim())

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

    The tree is the only USD asset: each marker's geometry, material, and
    metadata are authored directly below its marker child. Textures remain PNG
    files next to the tree because USD texture inputs are file assets. When a
    pose is supplied, it is authored directly on the marker child and is
    therefore restored whenever the tree is loaded. Without a pose, children
    are laid out from left to right with ``gap_mm`` clearance.
    """
    if not str(tree_path).strip():
        raise ValueError("Marker tree USD path cannot be empty.")
    if not math.isfinite(float(marker_length_mm)) or marker_length_mm <= 0:
        raise ValueError("Marker length must be greater than 0 mm.")
    if not math.isfinite(float(thickness_mm)) or thickness_mm <= 0:
        raise ValueError("Board thickness must be greater than 0 mm.")
    if not math.isfinite(float(board_to_marker_ratio)) or board_to_marker_ratio < 1.0:
        raise ValueError("Board-to-marker ratio must be at least 1.0.")
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
        if not root.IsValid() and not any(tree_stage.TraverseAll()):
            # Recover an empty crate left behind by an interrupted first write.
            UsdGeom.SetStageMetersPerUnit(tree_stage, 1.0)
            UsdGeom.SetStageUpAxis(tree_stage, UsdGeom.Tokens.z)
            root = UsdGeom.Xform.Define(tree_stage, "/ArUcoMarkerTree").GetPrim()
            tree_stage.SetDefaultPrim(root)
            root.SetCustomDataByKey("aruco:assetType", "markerTree")
            UsdGeom.Xform.Define(tree_stage, "/ArUcoMarkerTree/Markers")
        elif not root.IsValid() or root.GetPath().pathString != "/ArUcoMarkerTree":
            raise ValueError(f"Existing USD is not an ArUco marker tree: {tree_path}")
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
        markers = UsdGeom.Xform.Define(tree_stage, "/ArUcoMarkerTree/Markers").GetPrim()
    children = list(markers.GetChildren())

    requested_identity = (dictionary_name, int(marker_id))
    seen_identities: dict[tuple[str, int], str] = {}
    for child in children:
        child_dictionary = child.GetCustomDataByKey("aruco:dictionary")
        child_marker_id = child.GetCustomDataByKey("aruco:markerId")
        if child_dictionary is None or child_marker_id is None:
            continue
        child_identity = (str(child_dictionary), int(child_marker_id))
        if child_identity in seen_identities:
            raise ValueError(
                "Marker tree already contains an ambiguous duplicate identity: "
                f"{child_identity[0]} ID {child_identity[1]}."
            )
        seen_identities[child_identity] = child.GetPath().pathString
    if requested_identity in seen_identities:
        raise ValueError(
            "This marker identity is already present in the tree: "
            f"{dictionary_name} ID {int(marker_id)}."
        )

    safe_dictionary = _marker_dictionary_token(dictionary_name)
    base_name = f"Marker_{safe_dictionary}_{int(marker_id)}"
    marker_name = base_name
    suffix = 2
    while tree_stage.GetPrimAtPath(
        f"/ArUcoMarkerTree/Markers/{marker_name}"
    ).IsValid():
        marker_name = f"{base_name}_{suffix}"
        suffix += 1

    safe_tree_stem = re.sub(r"[^A-Za-z0-9_-]", "_", tree_path.stem)
    texture_path = tree_path.parent / "textures" / (
        f"{safe_tree_stem}_{marker_name}.png"
    )
    # Validate before writing the texture or saving the tree layer.
    cv2 = _load_cv2()
    _validate_marker_id(_dictionary(cv2, dictionary_name), int(marker_id))
    generate_marker_image(
        dictionary_name, int(marker_id), texture_path, texture_pixels
    )

    marker_length_mm = float(marker_length_mm)
    new_board_width_mm = marker_length_mm * float(board_to_marker_ratio)

    if position_m is not None:
        center_x_mm = float(position_m[0]) * 1000.0
    elif children:
        right_edges = []
        for child in children:
            center_x_mm = float(child.GetCustomDataByKey("aruco:centerXmm") or 0.0)
            width_mm = float(child.GetCustomDataByKey("aruco:boardSideLengthMm") or 0.0)
            right_edges.append(center_x_mm + width_mm / 2.0)
        center_x_mm = max(right_edges) + float(gap_mm) + new_board_width_mm / 2.0
    else:
        center_x_mm = 0.0

    marker_prim_path = f"/ArUcoMarkerTree/Markers/{marker_name}"
    marker_xform = UsdGeom.Xform.Define(tree_stage, marker_prim_path)
    relative_texture = texture_path.relative_to(tree_path.parent).as_posix()
    _author_board_in_stage(
        tree_stage,
        marker_prim_path,
        dictionary_name,
        int(marker_id),
        marker_length_mm,
        float(thickness_mm),
        float(board_to_marker_ratio),
        relative_texture,
    )
    if position_m is None:
        marker_position = Gf.Vec3d(center_x_mm / 1000.0, 0.0, 0.0)
    else:
        marker_position = Gf.Vec3d(*(float(value) for value in position_m))
    marker_xform.AddTranslateOp().Set(marker_position)
    if orientation_wxyz is not None:
        real, x, y, z = (float(value) for value in orientation_wxyz)
        marker_orientation = Gf.Quatd(real, Gf.Vec3d(x, y, z)).GetNormalized()
        marker_xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            marker_orientation
        )
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
        texture_path=texture_path,
        marker_prim_path=marker_prim_path,
        marker_count=marker_count,
    )


def _open_marker_tree(tree_path: str | Path) -> tuple[Path, Any]:
    """Open and validate an existing marker tree."""
    try:
        from pxr import Usd
    except ImportError as exc:
        raise RuntimeError(
            "Pixar USD Python modules are required; run this generator with Isaac "
            "Sim's Python interpreter."
        ) from exc

    resolved_path = Path(tree_path).expanduser().resolve()
    if resolved_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        resolved_path = resolved_path.with_suffix(".usd")
    if not resolved_path.exists():
        raise FileNotFoundError(f"Marker tree does not exist: {resolved_path}")
    stage = Usd.Stage.Open(str(resolved_path))
    root = stage.GetDefaultPrim() if stage else None
    if (
        root is None
        or not root.IsValid()
        or root.GetPath().pathString != "/ArUcoMarkerTree"
    ):
        raise ValueError(f"USD is not an ArUco marker tree: {resolved_path}")
    return resolved_path, stage


def _get_marker_prim(stage: Any, marker_prim_path: str) -> Any:
    """Return a direct child marker and reject arbitrary USD paths."""
    try:
        from pxr import Sdf
    except ImportError as exc:
        raise RuntimeError(
            "Pixar USD Python modules are required; run this generator with Isaac "
            "Sim's Python interpreter."
        ) from exc

    path = Sdf.Path(marker_prim_path)
    if path.GetParentPath() != Sdf.Path("/ArUcoMarkerTree/Markers"):
        raise ValueError(f"Path is not a marker tree child: {marker_prim_path}")
    marker = stage.GetPrimAtPath(path)
    if not marker.IsValid():
        raise ValueError(f"Marker does not exist in tree: {marker_prim_path}")
    return marker


def _marker_texture_path(tree_path: Path, marker: Any) -> Path | None:
    """Resolve a marker's local texture without accepting paths outside textures/."""
    try:
        from pxr import UsdShade
    except ImportError as exc:
        raise RuntimeError(
            "Pixar USD Python modules are required; run this generator with Isaac "
            "Sim's Python interpreter."
        ) from exc

    shader = UsdShade.Shader.Get(
        marker.GetStage(), marker.GetPath().AppendPath("MarkerMaterial/MarkerTexture")
    )
    asset = shader.GetInput("file").Get() if shader else None
    asset_path = getattr(asset, "path", "")
    if not asset_path:
        return None
    candidate = (tree_path.parent / asset_path).resolve()
    texture_root = (tree_path.parent / "textures").resolve()
    if not candidate.is_relative_to(texture_root) or candidate.suffix.lower() != ".png":
        return None
    return candidate


def save_marker_transform_to_tree_usd(
    tree_path: str | Path,
    marker_prim_path: str,
    local_transform: Any,
) -> None:
    """Persist one composed marker transform directly in the tree USD."""
    try:
        from pxr import Gf, UsdGeom
    except ImportError as exc:
        raise RuntimeError(
            "Pixar USD Python modules are required; run this generator with Isaac "
            "Sim's Python interpreter."
        ) from exc

    resolved_path, stage = _open_marker_tree(tree_path)
    marker = _get_marker_prim(stage, marker_prim_path)
    matrix = Gf.Matrix4d(local_transform)
    xformable = UsdGeom.Xformable(marker)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(matrix)
    marker.SetCustomDataByKey(
        "aruco:centerXmm", float(matrix.ExtractTranslation()[0]) * 1000.0
    )
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"Failed to save marker tree: {resolved_path}")


def update_marker_in_tree_usd(
    tree_path: str | Path,
    marker_prim_path: str,
    dictionary_name: str,
    marker_id: int,
    marker_length_mm: float,
    *,
    local_transform: Any | None = None,
    thickness_mm: float = 1.0,
    texture_pixels: int = 1024,
    board_to_marker_ratio: float = 1.05,
) -> GeneratedMarkerTree:
    """Update identity, size, geometry, texture, and optionally pose in the tree."""
    if not math.isfinite(float(marker_length_mm)) or marker_length_mm <= 0:
        raise ValueError("Marker length must be greater than 0 mm.")
    if not math.isfinite(float(thickness_mm)) or thickness_mm <= 0:
        raise ValueError("Board thickness must be greater than 0 mm.")
    if not math.isfinite(float(board_to_marker_ratio)) or board_to_marker_ratio < 1.0:
        raise ValueError("Board-to-marker ratio must be at least 1.0.")

    try:
        from pxr import Gf, Sdf, UsdGeom
    except ImportError as exc:
        raise RuntimeError(
            "Pixar USD Python modules are required; run this generator with Isaac "
            "Sim's Python interpreter."
        ) from exc

    cv2 = _load_cv2()
    _validate_marker_id(_dictionary(cv2, dictionary_name), int(marker_id))
    resolved_path, stage = _open_marker_tree(tree_path)
    marker = _get_marker_prim(stage, marker_prim_path)
    old_texture_path = _marker_texture_path(resolved_path, marker)

    requested_identity = (dictionary_name, int(marker_id))
    markers = stage.GetPrimAtPath("/ArUcoMarkerTree/Markers")
    for sibling in markers.GetChildren():
        if sibling == marker:
            continue
        sibling_identity = (
            str(sibling.GetCustomDataByKey("aruco:dictionary")),
            int(sibling.GetCustomDataByKey("aruco:markerId")),
        )
        if sibling_identity == requested_identity:
            raise ValueError(
                "This marker identity is already present in the tree: "
                f"{dictionary_name} ID {int(marker_id)}."
            )

    safe_dictionary = _marker_dictionary_token(dictionary_name)
    base_name = f"Marker_{safe_dictionary}_{int(marker_id)}"
    marker_name = base_name
    suffix = 2
    old_path = marker.GetPath()
    while (
        stage.GetPrimAtPath(f"/ArUcoMarkerTree/Markers/{marker_name}").IsValid()
        and Sdf.Path(f"/ArUcoMarkerTree/Markers/{marker_name}") != old_path
    ):
        marker_name = f"{base_name}_{suffix}"
        suffix += 1

    new_path = Sdf.Path(f"/ArUcoMarkerTree/Markers/{marker_name}")
    if new_path != old_path:
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(Sdf.NamespaceEdit.Rename(old_path, marker_name))
        if not stage.GetRootLayer().Apply(edit):
            raise RuntimeError(f"Could not rename marker Prim to {new_path}")
        marker = stage.GetPrimAtPath(new_path)

    if local_transform is not None:
        matrix = Gf.Matrix4d(local_transform)
        xformable = UsdGeom.Xformable(marker)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(matrix)
        marker.SetCustomDataByKey(
            "aruco:centerXmm", float(matrix.ExtractTranslation()[0]) * 1000.0
        )

    for child_name in ("Backing", "Marker", "MarkerMaterial"):
        stage.RemovePrim(marker.GetPath().AppendChild(child_name))

    safe_tree_stem = re.sub(r"[^A-Za-z0-9_-]", "_", resolved_path.stem)
    texture_path = resolved_path.parent / "textures" / (
        f"{safe_tree_stem}_{marker_name}.png"
    )
    generate_marker_image(
        dictionary_name, int(marker_id), texture_path, texture_pixels
    )
    relative_texture = texture_path.relative_to(resolved_path.parent).as_posix()
    _author_board_in_stage(
        stage,
        marker.GetPath().pathString,
        dictionary_name,
        int(marker_id),
        float(marker_length_mm),
        float(thickness_mm),
        float(board_to_marker_ratio),
        relative_texture,
    )
    marker.SetCustomDataByKey(
        "aruco:boardSideLengthMm",
        float(marker_length_mm) * float(board_to_marker_ratio),
    )

    root = stage.GetDefaultPrim()
    marker_count = len(list(markers.GetChildren()))
    root.SetCustomDataByKey("aruco:markerCount", marker_count)
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"Failed to save marker tree: {resolved_path}")
    if old_texture_path and old_texture_path != texture_path and old_texture_path.exists():
        old_texture_path.unlink()

    return GeneratedMarkerTree(
        tree_path=resolved_path,
        texture_path=texture_path,
        marker_prim_path=new_path.pathString,
        marker_count=marker_count,
    )


def delete_marker_from_tree_usd(
    tree_path: str | Path,
    marker_prim_path: str,
) -> int:
    """Delete one marker Prim and its managed texture from the tree USD."""
    resolved_path, stage = _open_marker_tree(tree_path)
    marker = _get_marker_prim(stage, marker_prim_path)
    texture_path = _marker_texture_path(resolved_path, marker)
    stage.RemovePrim(marker.GetPath())

    markers = stage.GetPrimAtPath("/ArUcoMarkerTree/Markers")
    children = list(markers.GetChildren())
    for index, child in enumerate(children):
        child.SetCustomDataByKey("aruco:index", index)
    marker_count = len(children)
    stage.GetDefaultPrim().SetCustomDataByKey("aruco:markerCount", marker_count)
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"Failed to save marker tree: {resolved_path}")
    if texture_path and texture_path.exists():
        texture_path.unlink()
    return marker_count


def read_marker_tree_usd(tree_path: str | Path) -> tuple[MarkerTreeEntry, ...]:
    """Read the complete marker registry from a tree USD in child order."""
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError(
            "Pixar USD Python modules are required; run this generator with Isaac "
            "Sim's Python interpreter."
        ) from exc

    _, stage = _open_marker_tree(tree_path)
    markers = stage.GetPrimAtPath("/ArUcoMarkerTree/Markers")
    if not markers.IsValid():
        return ()

    entries = []
    identities = set()
    for marker in markers.GetChildren():
        dictionary_name = marker.GetCustomDataByKey("aruco:dictionary")
        marker_id = marker.GetCustomDataByKey("aruco:markerId")
        marker_length_mm = marker.GetCustomDataByKey("aruco:markerLengthMm")
        if dictionary_name is None or marker_id is None or marker_length_mm is None:
            raise ValueError(
                f"Marker is missing required ArUco metadata: {marker.GetPath()}"
            )
        identity = (str(dictionary_name), int(marker_id))
        if identity in identities:
            raise ValueError(
                "Marker tree contains duplicate detection identity: "
                f"{identity[0]} ID {identity[1]}."
            )
        identities.add(identity)
        entries.append(
            MarkerTreeEntry(
                prim_path=marker.GetPath().pathString,
                dictionary_name=identity[0],
                marker_id=identity[1],
                marker_length_mm=float(marker_length_mm),
                local_transform=UsdGeom.Xformable(marker).GetLocalTransformation(
                    Usd.TimeCode.Default()
                ),
            )
        )
    return tuple(entries)
