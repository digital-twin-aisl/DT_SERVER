"""Build one explicit, world-space inference workspace for each edge.

The deployment manifest owns scene assets, camera assignment, and optional
edge AOIs.  Model code consumes the resulting :class:`EdgeWorkspace` instead
of reconstructing scene geometry from unrelated configuration fields.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .ground import GroundSurface


WORKSPACE_FORMAT_VERSION = 1
V2V_SIZE_MULTIPLE = 4


def calculate_cube_size(
    size_mm: Sequence[float],
    reference_size_mm: Sequence[float],
    reference_cube_size: Sequence[int],
    *,
    multiple: int = V2V_SIZE_MULTIPLE,
) -> np.ndarray:
    """Preserve trained mm/voxel while producing a V2V-compatible shape."""

    size = np.asarray(size_mm, dtype=np.float64)
    reference_size = np.asarray(reference_size_mm, dtype=np.float64)
    reference_bins = np.asarray(reference_cube_size, dtype=np.int64)
    if (
        size.shape != (3,)
        or reference_size.shape != (3,)
        or reference_bins.shape != (3,)
        or not np.isfinite(size).all()
        or not np.isfinite(reference_size).all()
        or np.any(size <= 0)
        or np.any(reference_size <= 0)
        or np.any(reference_bins < 2)
        or multiple < 1
    ):
        raise ValueError("cube sizes must be finite positive XYZ triplets")

    reference_spacing = reference_size / (reference_bins - 1)
    ideal_bins = size / reference_spacing + 1.0
    bins = np.empty(3, dtype=np.int64)
    bins[:2] = np.ceil(ideal_bins[:2] / multiple).astype(np.int64) * multiple
    bins[2] = max(
        multiple,
        int(np.floor(ideal_bins[2] / multiple + 0.5)) * multiple,
    )
    return bins


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the counter-clockwise convex hull of finite XY points."""

    ordered = sorted(tuple(point) for point in points)

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def fit_oriented_rectangle(
    points_xy_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a minimum-area containing rectangle around an XY point set."""

    points = np.asarray(points_xy_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("points_xy_mm must contain at least three XY positions")
    if not np.isfinite(points).all():
        raise ValueError("points_xy_mm must contain only finite values")
    points = np.unique(points, axis=0)
    hull = convex_hull(points)
    if len(hull) < 3:
        raise ValueError("points_xy_mm must span a non-degenerate XY region")

    best = None
    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        edge_length = np.linalg.norm(edge)
        if edge_length <= 1e-6:
            continue
        axis_x = edge / edge_length
        axis_y = np.asarray([-axis_x[1], axis_x[0]])
        axes = np.stack([axis_x, axis_y])
        local = points @ axes.T
        local_min = local.min(axis=0)
        local_max = local.max(axis=0)
        size = local_max - local_min
        candidate = (float(np.prod(size)), axes, local_min, local_max)
        if best is None or candidate[0] < best[0] - 1e-6:
            best = candidate

    if best is None:
        raise ValueError("could not fit a rectangle to XY points")
    _, axes, local_min, local_max = best
    size = local_max - local_min
    if size[0] < size[1]:
        axes = np.stack([axes[1], -axes[0]])
        local = points @ axes.T
        local_min = local.min(axis=0)
        local_max = local.max(axis=0)
        size = local_max - local_min
    if axes[0, 0] < 0 or (abs(axes[0, 0]) <= 1e-12 and axes[0, 1] < 0):
        axes = -axes
        local = points @ axes.T
        local_min = local.min(axis=0)
        local_max = local.max(axis=0)
        size = local_max - local_min
    center = ((local_min + local_max) / 2.0) @ axes
    return center, size, axes


def fit_camera_rectangle(
    camera_xy_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility fallback for a four-camera containing rectangle."""

    points = np.asarray(camera_xy_mm, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("camera_xy_mm must contain four finite XY positions")
    if len(convex_hull(points)) != 4:
        raise ValueError(
            "the four camera positions must form a non-degenerate quadrilateral"
        )
    return fit_oriented_rectangle(points)


def cuboid_xy_grid(
    center_xy_mm: Sequence[float],
    size_xy_mm: Sequence[float],
    xy_axes: np.ndarray,
    cube_size_xy: Sequence[int],
) -> np.ndarray:
    """Return a regular local XY lattice transformed to world coordinates."""

    size = np.asarray(size_xy_mm, dtype=np.float64)
    bins = np.asarray(cube_size_xy, dtype=np.int64)
    if size.shape != (2,) or bins.shape != (2,) or np.any(bins < 2):
        raise ValueError("XY size and bin counts must each contain two values")
    local_x = np.linspace(-size[0] / 2.0, size[0] / 2.0, int(bins[0]))
    local_y = np.linspace(-size[1] / 2.0, size[1] / 2.0, int(bins[1]))
    grid_x, grid_y = np.meshgrid(local_x, local_y, indexing="ij")
    local_xy = np.stack([grid_x, grid_y], axis=-1)
    return local_xy @ np.asarray(xy_axes, dtype=np.float64) + np.asarray(
        center_xy_mm,
        dtype=np.float64,
    )


def query_ground_heights_mm(
    surface: Any,
    xy_mm: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Query a GroundSurface without one huge point/triangle allocation."""

    points = np.asarray(xy_mm, dtype=np.float64)
    flat_points = points.reshape(-1, 2)
    chunks = [
        surface.heights_mm(flat_points[start : start + chunk_size])
        for start in range(0, len(flat_points), chunk_size)
    ]
    return np.concatenate(chunks).reshape(points.shape[:-1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(base_path: Path, value: str, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} not found: {path}")
    return path


def _pair_mm(value: Any, field: str) -> tuple[float, float]:
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{field} must contain two finite values")
    if values[1] <= values[0]:
        raise ValueError(f"{field} maximum must be greater than its minimum")
    return float(values[0]), float(values[1])


@dataclass(frozen=True, slots=True)
class WorkspacePolicy:
    """Deployment policy used while deriving an edge workspace."""

    strategy: str = "visibility"
    min_views: int = 2
    sample_spacing_mm: float = 250.0
    boundary_padding_mm: float = 125.0
    root_height_mm: float = 900.0
    root_clearance_mm: tuple[float, float] = (400.0, 1400.0)
    foot_clearance_mm: tuple[float, float] = (-150.0, 350.0)
    preserve_voxel_spacing: bool = True
    camera_rectangle_fallback: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "WorkspacePolicy":
        data = dict(value or {})
        policy = cls(
            strategy=str(data.get("strategy", "visibility")),
            min_views=int(data.get("min_views", 2)),
            sample_spacing_mm=float(data.get("sample_spacing_mm", 250.0)),
            boundary_padding_mm=float(data.get("boundary_padding_mm", 125.0)),
            root_height_mm=float(data.get("root_height_mm", 900.0)),
            root_clearance_mm=_pair_mm(
                data.get("root_clearance_mm", [400.0, 1400.0]),
                "workspace.root_clearance_mm",
            ),
            foot_clearance_mm=_pair_mm(
                data.get("foot_clearance_mm", [-150.0, 350.0]),
                "workspace.foot_clearance_mm",
            ),
            preserve_voxel_spacing=bool(
                data.get("preserve_voxel_spacing", True)
            ),
            camera_rectangle_fallback=bool(
                data.get("camera_rectangle_fallback", True)
            ),
        )
        if policy.strategy not in {"visibility", "explicit_aoi"}:
            raise ValueError(
                "workspace.strategy must be 'visibility' or 'explicit_aoi'"
            )
        if policy.min_views < 1:
            raise ValueError("workspace.min_views must be at least 1")
        if policy.sample_spacing_mm <= 0:
            raise ValueError("workspace.sample_spacing_mm must be positive")
        if policy.boundary_padding_mm < 0:
            raise ValueError("workspace.boundary_padding_mm must not be negative")
        if not np.isfinite(
            [
                policy.sample_spacing_mm,
                policy.boundary_padding_mm,
                policy.root_height_mm,
            ]
        ).all():
            raise ValueError("workspace policy values must be finite")
        return policy


@dataclass(frozen=True, slots=True)
class SpatialContext:
    """Shared world geometry and policy loaded from one deployment manifest."""

    scene_id: str
    context_id: str
    calibration_sha256: str
    calibration_path: Path
    world_origin_m: tuple[float, float, float]
    ground_usd_path: Path
    ground_cache_path: Path | None
    ground_surface: GroundSurface
    policy: WorkspacePolicy
    edge_camera_ids: dict[str, tuple[int, ...]]
    edge_aoi_xy_mm: dict[str, np.ndarray]

    def identity(self) -> dict[str, str]:
        return {
            "scene_id": self.scene_id,
            "context_id": self.context_id,
            "calibration_sha256": self.calibration_sha256,
        }


@dataclass(frozen=True, slots=True)
class EdgeWorkspace:
    """The complete dense tensor domain and physical validity mask for one edge."""

    edge_id: str
    source: str
    center_mm: np.ndarray
    size_mm: np.ndarray
    cube_size: np.ndarray
    xy_axes: np.ndarray
    grid_to_world: np.ndarray
    footprint_xy_mm: np.ndarray
    ground_height_mm: np.ndarray
    valid_mask: np.ndarray
    min_views: int
    root_clearance_mm: tuple[float, float]
    workspace_id: str

    @property
    def voxel_spacing_mm(self) -> np.ndarray:
        return self.size_mm / (self.cube_size - 1)


def load_spatial_context(manifest_path: str | Path) -> SpatialContext | None:
    """Load scene geometry from a deployment manifest.

    Legacy edge metadata without a ``scene`` object intentionally returns
    ``None`` so old flat-origin deployments remain usable.
    """

    path = Path(manifest_path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    scene = manifest.get("scene")
    if scene is None:
        return None
    if not isinstance(scene, dict):
        raise ValueError("deployment.scene must be an object")

    ground_usd = _resolve_path(
        path,
        str(scene.get("ground_usd", "")),
        "scene.ground_usd",
    )
    cache_value = str(scene.get("ground_cache", "")).strip()
    ground_cache = (
        _resolve_path(path, cache_value, "scene.ground_cache")
        if cache_value
        else None
    )
    if ground_cache is None:
        surface = GroundSurface.from_usd(ground_usd)
    else:
        surface = GroundSurface.from_cache(ground_cache, ground_usd)

    calibration_path = _resolve_path(
        path,
        str(manifest.get("calibration_result", "")),
        "calibration_result",
    )
    policy_data = manifest.get("workspace")
    if policy_data is not None and not isinstance(policy_data, dict):
        raise ValueError("deployment.workspace must be an object")
    policy = WorkspacePolicy.from_mapping(policy_data)
    world_origin = np.asarray(
        manifest.get("world_origin_m", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if world_origin.shape != (3,) or not np.isfinite(world_origin).all():
        raise ValueError("world_origin_m must contain three finite values")

    edge_camera_ids: dict[str, tuple[int, ...]] = {}
    edge_aoi_xy_mm: dict[str, np.ndarray] = {}
    for edge in manifest.get("edges") or []:
        edge_id = str(edge["id"])
        camera_ids = edge.get("camera_ids") or edge.get("cameras") or []
        edge_camera_ids[edge_id] = tuple(int(value) for value in camera_ids)
        workspace = edge.get("workspace") or {}
        polygon_m = workspace.get("polygon_xy_m")
        if polygon_m is not None:
            polygon = np.asarray(polygon_m, dtype=np.float64)
            if (
                polygon.ndim != 2
                or polygon.shape[1] != 2
                or len(polygon) < 3
                or not np.isfinite(polygon).all()
            ):
                raise ValueError(
                    f"{edge_id}.workspace.polygon_xy_m must contain at least "
                    "three finite XY points"
                )
            edge_aoi_xy_mm[edge_id] = polygon * 1000.0

    context_payload = {
        "workspace_format_version": WORKSPACE_FORMAT_VERSION,
        "scene": scene,
        "workspace": policy_data or {},
        "edges": [
            {
                "id": edge.get("id"),
                "camera_ids": edge.get("camera_ids") or edge.get("cameras") or [],
                "workspace": edge.get("workspace") or {},
            }
            for edge in (manifest.get("edges") or [])
            if edge.get("enabled", True)
        ],
        "ground_sha256": _sha256(ground_usd),
    }
    context_id = hashlib.sha256(
        json.dumps(
            context_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return SpatialContext(
        scene_id=str(scene.get("id") or ground_usd.stem),
        context_id=context_id,
        calibration_sha256=_sha256(calibration_path),
        calibration_path=calibration_path,
        world_origin_m=tuple(float(value) for value in world_origin),
        ground_usd_path=ground_usd,
        ground_cache_path=ground_cache,
        ground_surface=surface,
        policy=policy,
        edge_camera_ids=edge_camera_ids,
        edge_aoi_xy_mm=edge_aoi_xy_mm,
    )


def _camera_visibility(
    points_mm: np.ndarray,
    camera: Mapping[str, Any],
    image_size: Sequence[int],
) -> np.ndarray:
    """Return image/depth visibility using the same radial model as VoxelPose."""

    points = np.asarray(points_mm, dtype=np.float64)
    rotation = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
    center = np.asarray(camera["T"], dtype=np.float64).reshape(3)
    camera_points = (points - center) @ rotation.T
    depth = camera_points[:, 2]
    normalized = camera_points[:, :2] / (depth[:, None] + 1e-5)

    radial_coefficients = np.asarray(camera["k"], dtype=np.float64).reshape(3)
    tangential = np.asarray(camera["p"], dtype=np.float64).reshape(2)
    radius2 = np.sum(normalized**2, axis=1)
    radius2 = np.minimum(radius2, 1e10)
    radial = (
        1.0
        + radial_coefficients[0] * radius2
        + radial_coefficients[1] * radius2**2
        + radial_coefficients[2] * radius2**3
    )
    tangent = tangential[0] * normalized[:, 1] + tangential[1] * normalized[:, 0]
    corrected = normalized * (radial + 2.0 * tangent)[:, None]
    corrected += np.column_stack(
        [tangential[1] * radius2, tangential[0] * radius2]
    )
    pixels = corrected * np.asarray(
        [camera["fx"], camera["fy"]], dtype=np.float64
    ) + np.asarray([camera["cx"], camera["cy"]], dtype=np.float64)
    width, height = (int(value) for value in image_size)
    return (
        (depth > 0.0)
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )


def _visibility_count(
    points_mm: np.ndarray,
    cameras: Sequence[Mapping[str, Any]],
    image_size: Sequence[int],
) -> np.ndarray:
    count = np.zeros(len(points_mm), dtype=np.uint8)
    for camera in cameras:
        count += _camera_visibility(points_mm, camera, image_size)
    return count


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep one 4-connected ground region so an edge gets one coherent grid."""

    if mask.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    visited = np.zeros(mask.shape, dtype=np.bool_)
    largest: list[tuple[int, int]] = []
    width, height = mask.shape
    for start_x, start_y in np.argwhere(mask):
        start = int(start_x), int(start_y)
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        component = []
        while queue:
            x, y = queue.popleft()
            component.append((x, y))
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and mask[nx, ny]
                    and not visited[nx, ny]
                ):
                    visited[nx, ny] = True
                    queue.append((nx, ny))
        if len(component) > len(largest):
            largest = component

    result = np.zeros(mask.shape, dtype=np.bool_)
    if largest:
        indices = np.asarray(largest, dtype=np.int64)
        result[indices[:, 0], indices[:, 1]] = True
    return result


def _points_in_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    """Vectorized even/odd containment with polygon boundaries included."""

    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    inside = np.zeros(len(points), dtype=np.bool_)
    x = points[:, 0]
    y = points[:, 1]
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > y) != (y2 > y)
        intersection_x = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= crosses & (x <= intersection_x)
        segment = current - previous
        relative = points - previous
        cross = np.abs(segment[0] * relative[:, 1] - segment[1] * relative[:, 0])
        dot = relative @ segment
        on_boundary = (
            (cross <= 1e-5)
            & (dot >= -1e-5)
            & (dot <= float(segment @ segment) + 1e-5)
        )
        inside |= on_boundary
        previous = current
    return inside


def _automatic_visibility_aoi(
    surface: GroundSurface,
    cameras: Sequence[Mapping[str, Any]],
    image_size: Sequence[int],
    policy: WorkspacePolicy,
) -> np.ndarray:
    triangle_xy = surface.triangles_mm[:, :, :2]
    minimum = triangle_xy.min(axis=(0, 1))
    maximum = triangle_xy.max(axis=(0, 1))
    spacing = policy.sample_spacing_mm
    x_values = np.arange(minimum[0] + spacing / 2.0, maximum[0], spacing)
    y_values = np.arange(minimum[1] + spacing / 2.0, maximum[1], spacing)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="ij")
    grid_xy = np.stack([grid_x, grid_y], axis=-1)
    ground_height = query_ground_heights_mm(surface, grid_xy)
    points = np.column_stack(
        [
            grid_xy.reshape(-1, 2),
            ground_height.reshape(-1) + policy.root_height_mm,
        ]
    )
    visible = np.isfinite(points[:, 2]) & (
        _visibility_count(points, cameras, image_size) >= policy.min_views
    )
    component = _largest_component(visible.reshape(grid_xy.shape[:2]))
    selected = grid_xy[component]
    if len(selected) < 3:
        raise ValueError("camera frusta do not define a connected Ground workspace")
    return selected


def _grid_to_world(center_mm: np.ndarray, xy_axes: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = xy_axes[0, 0]
    transform[0, 1] = xy_axes[1, 0]
    transform[1, 0] = xy_axes[0, 1]
    transform[1, 1] = xy_axes[1, 1]
    transform[:3, 3] = center_mm
    return transform


def _workspace_digest(
    edge_id: str,
    center_mm: np.ndarray,
    size_mm: np.ndarray,
    cube_size: np.ndarray,
    xy_axes: np.ndarray,
    valid_mask: np.ndarray,
) -> str:
    digest = hashlib.sha256(edge_id.encode("utf-8"))
    for value in (center_mm, size_mm, cube_size, xy_axes):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(np.packbits(valid_mask.reshape(-1)).tobytes())
    return digest.hexdigest()


def build_edge_workspace(
    context: SpatialContext,
    edge_id: str,
    cameras: Sequence[Mapping[str, Any]],
    image_size: Sequence[int],
    reference_size_mm: Sequence[float],
    reference_cube_size: Sequence[int],
    *,
    auto_cube_size: bool | None = None,
) -> EdgeWorkspace:
    """Resolve an edge AOI into one oriented dense grid plus physical mask."""

    if not cameras:
        raise ValueError(f"{edge_id} has no calibrated cameras")
    policy = context.policy
    if auto_cube_size is None:
        auto_cube_size = policy.preserve_voxel_spacing
    if policy.min_views > len(cameras):
        raise ValueError(
            f"workspace.min_views={policy.min_views} exceeds {len(cameras)} "
            f"cameras for {edge_id}"
        )
    expected_ids = context.edge_camera_ids.get(edge_id)
    received_ids = tuple(int(camera["id"]) for camera in cameras)
    if expected_ids is not None and received_ids != expected_ids:
        raise ValueError(
            f"camera order mismatch for {edge_id}: expected {expected_ids}, "
            f"got {received_ids}"
        )

    explicit_aoi = context.edge_aoi_xy_mm.get(edge_id)
    if explicit_aoi is not None:
        source = "explicit_aoi"
        footprint = explicit_aoi
        center_xy, size_xy, xy_axes = fit_oriented_rectangle(footprint)
    else:
        if policy.strategy == "explicit_aoi":
            raise ValueError(
                f"{edge_id} has no workspace.polygon_xy_m in an "
                "explicit_aoi deployment"
            )
        try:
            visible_points = _automatic_visibility_aoi(
                context.ground_surface,
                cameras,
                image_size,
                policy,
            )
            source = "visibility"
            footprint = convex_hull(visible_points)
            center_xy, size_xy, xy_axes = fit_oriented_rectangle(visible_points)
            size_xy = size_xy + 2.0 * policy.boundary_padding_mm
        except ValueError:
            if not policy.camera_rectangle_fallback or len(cameras) != 4:
                raise
            source = "camera_rectangle_fallback"
            camera_xy = np.stack(
                [
                    np.asarray(camera["T"], dtype=np.float64).reshape(3)[:2]
                    for camera in cameras
                ]
            )
            center_xy, size_xy, xy_axes = fit_camera_rectangle(camera_xy)
            footprint = convex_hull(camera_xy)

    reference_size = np.asarray(reference_size_mm, dtype=np.float64)
    reference_bins = np.asarray(reference_cube_size, dtype=np.int64)
    provisional_size = np.asarray([size_xy[0], size_xy[1], reference_size[2]])
    xy_bins = (
        calculate_cube_size(provisional_size, reference_size, reference_bins)[:2]
        if auto_cube_size
        else reference_bins[:2]
    )
    xy_grid = cuboid_xy_grid(center_xy, size_xy, xy_axes, xy_bins)
    ground_height = query_ground_heights_mm(context.ground_surface, xy_grid)
    footprint_valid = _points_in_polygon(
        xy_grid.reshape(-1, 2),
        footprint,
    ).reshape(xy_grid.shape[:2])
    usable_ground = ground_height[footprint_valid & np.isfinite(ground_height)]
    if len(usable_ground) == 0:
        raise ValueError(f"Ground does not overlap the {edge_id} workspace")

    clearance_min, clearance_max = policy.root_clearance_mm
    z_min = float(usable_ground.min()) + clearance_min
    z_max = float(usable_ground.max()) + clearance_max
    size = np.asarray([size_xy[0], size_xy[1], z_max - z_min])
    center = np.asarray([center_xy[0], center_xy[1], (z_min + z_max) / 2.0])
    cube_size = (
        calculate_cube_size(size, reference_size, reference_bins)
        if auto_cube_size
        else reference_bins.copy()
    )

    if not np.array_equal(cube_size[:2], xy_bins):
        xy_grid = cuboid_xy_grid(center_xy, size_xy, xy_axes, cube_size[:2])
        ground_height = query_ground_heights_mm(context.ground_surface, xy_grid)
        footprint_valid = _points_in_polygon(
            xy_grid.reshape(-1, 2),
            footprint,
        ).reshape(xy_grid.shape[:2])

    z_values = np.linspace(z_min, z_max, int(cube_size[2]))
    clearance = z_values[None, None, :] - ground_height[:, :, None]
    terrain_valid = (
        footprint_valid[:, :, None]
        & np.isfinite(clearance)
        & (clearance >= clearance_min)
        & (clearance <= clearance_max)
    )
    world_points = np.empty((*terrain_valid.shape, 3), dtype=np.float64)
    world_points[:, :, :, :2] = xy_grid[:, :, None, :]
    world_points[:, :, :, 2] = z_values[None, None, :]
    visible_count = _visibility_count(
        world_points.reshape(-1, 3),
        cameras,
        image_size,
    ).reshape(terrain_valid.shape)
    valid_mask = terrain_valid & (visible_count >= policy.min_views)
    if not valid_mask.any():
        raise ValueError(f"{edge_id} workspace has no valid multi-view voxels")

    workspace_id = _workspace_digest(
        edge_id,
        center,
        size,
        cube_size,
        xy_axes,
        valid_mask,
    )
    return EdgeWorkspace(
        edge_id=edge_id,
        source=source,
        center_mm=center,
        size_mm=size,
        cube_size=cube_size,
        xy_axes=xy_axes,
        grid_to_world=_grid_to_world(center, xy_axes),
        footprint_xy_mm=np.asarray(footprint, dtype=np.float64),
        ground_height_mm=ground_height,
        valid_mask=valid_mask,
        min_views=policy.min_views,
        root_clearance_mm=policy.root_clearance_mm,
        workspace_id=workspace_id,
    )
