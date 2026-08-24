# ruff: noqa: E402

import argparse
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

SERVER_WORKER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_WORKER_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import apps  # noqa: E402,F401
from dt_common.spatial.workspace import build_edge_workspace, load_spatial_context
from apps.server_worker.config.config import config as focus_config
from apps.server_worker.config.config import update_config as update_focus_config
from apps.server_worker.priority_engine import PriorityConfig, PriorityEngine
from apps.server_worker.src.pose.core.config import config as sp3d_config
from apps.server_worker.src.pose.core.config import (
    update_config as update_sp3d_config,
)
from apps.server_worker.src.pose.models.multi_person_posenet_ssv import (
    get_multi_person_pose_net,
)
from apps.server_worker.src.pose.utils import cameras
from apps.server_worker.src.protocol.scene_zenoh import (
    DEFAULT_SCENE_TOPIC,
    ZenohScenePublisher,
)
from apps.server_worker.src.protocol.zenoh import ZenohDataLoader
from apps.server_worker.src.protocol.zmq import Protocol
from apps.server_worker.src.reid.sliding_clustering import ClusteringSliding
from apps.server_worker.src.utils.edgemetadata import EdgeMetadataLoader
from apps.server_worker.src.utils.edgemetadata import DEFAULT_METADATA_PATH


logger = logging.getLogger(__name__)

LodLevel = Literal[0, 1, 2]
LodAssignments = dict[int, LodLevel]
LodAssignCallback = Callable[[list[int]], LodAssignments]
OUTPUT_SCHEMA_VERSION = 1
OUTPUT_COORDINATE_SYSTEM = {
    "frame": "USD world",
    "up_axis": "Z",
    "unit": "millimetre",
}
DEFAULT_MAX_OUTPUT_PEOPLE = 10
DEFAULT_LOD2_PEOPLE = 1
MAX_ROOT_REPROJECTION_ERROR_PX = 250.0
ROOT_POSITION_TO_METERS = 0.001

# ===== TEMP VISER DEBUG: delete this block and the marked call in main() =====
ENABLE_ZMQ_OUTPUT = False
ENABLE_VISER_DEBUG_OUTPUT = True
VISER_DEBUG_OUTPUT_PATH = SERVER_WORKER_DIR / "data" / "viser_scenes.jsonl"


def save_scene_for_viser(
    scene: "SceneOutput",
    output_path: Path = VISER_DEBUG_OUTPUT_PATH,
) -> None:
    """Append one scene as one JSONL record for Viser playback."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        scene.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
    )
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(serialized + "\n")


# ===== END TEMP VISER DEBUG =====


@dataclass(frozen=True, slots=True)
class RootCandidate:
    """한 Edge 안에서 Global ID와 연결된 root 후보."""

    global_id: int
    edge_id: str
    edge_index: int
    root_index: int
    confidence: float
    reprojection_error: float
    observation_count: int


@dataclass(frozen=True, slots=True)
class RootOutput:
    """USD world 좌표계의 millimetre 단위 사람 root."""

    edge_id: str
    candidate_index: int
    position: list[float]
    confidence: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "candidate_index": self.candidate_index,
            "position": self.position,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class PoseOutput:
    """root와 동일한 USD world millimetre 단위 3D joint 좌표."""

    joint_format: str
    joints: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_format": self.joint_format,
            "joints": self.joints,
        }


@dataclass(frozen=True, slots=True)
class PersonEntity:
    """서버 파이프라인이 출력하는 사람 단위 entity."""

    global_id: int
    lod: LodLevel
    root: RootOutput
    pose: PoseOutput | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_id": self.global_id,
            "lod": self.lod,
            "root": self.root.to_dict(),
            "pose": self.pose.to_dict() if self.pose is not None else None,
        }


@dataclass(frozen=True, slots=True)
class SceneOutput:
    """동기화된 한 시점의 전체 사람 entity 출력."""

    timestamp: float
    sync_spread_seconds: float
    people: list[PersonEntity]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "coordinate_system": OUTPUT_COORDINATE_SYSTEM,
            "timestamp": self.timestamp,
            "sync_spread_seconds": self.sync_spread_seconds,
            "people": [person.to_dict() for person in self.people],
        }


def setup_cuda() -> torch.device:
    """Configure CUDA when inference starts, not while this module is imported."""
    logger.info("[1/8] Initializing CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for server inference")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = sp3d_config.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = sp3d_config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = sp3d_config.CUDNN.DETERMINISTIC
    device = torch.device("cuda")
    logger.info("CUDA is ready: device=%s", device)
    return device


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch AISL Inference")
    parser.add_argument(
        "--cfg-focus",
        dest="cfg_focus",
        default=None,
        help="Focus configuration file",
    )
    parser.add_argument(
        "--pose-config",
        default=None,
        help="Pose model configuration file",
    )
    parser.add_argument(
        "--tensorrt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable TensorRT",
    )
    parser.add_argument(
        "--zenoh-endpoint",
        dest="zenoh_endpoint",
        type=str,
        default=None,
        help="Zenoh endpoint to connect to",
    )
    parser.add_argument(
        "--zenoh-config",
        dest="zenoh_config",
        type=str,
        default=None,
        help="Path to the Zenoh configuration file",
    )
    parser.add_argument(
        "--deployment",
        "--edge-metadata",
        dest="deployment",
        default=str(DEFAULT_METADATA_PATH),
        help="Shared scene, calibration, edge assignment, and workspace manifest",
    )
    parser.add_argument(
        "--edge-id",
        dest="edge_ids",
        action="append",
        help="Edge ID to consume; repeat it to override the enabled metadata edges",
    )
    parser.add_argument("--buffer-size", type=int, default=10)
    parser.add_argument("--sync-tolerance", type=float, default=0.03)
    parser.add_argument("--receive-timeout", type=float, default=1.0)
    parser.add_argument(
        "--max-output-people",
        type=int,
        default=DEFAULT_MAX_OUTPUT_PEOPLE,
        help="Maximum number of person entities emitted per synchronized scene",
    )
    parser.add_argument(
        "--lod2-count",
        type=int,
        default=DEFAULT_LOD2_PEOPLE,
        help="Number of highest-priority people assigned to LOD 2",
    )
    parser.add_argument(
        "--priority-hazard",
        action="append",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help=(
            "Hazard point in metre-based root world coordinates; may be repeated. "
            "Defaults to the PriorityEngine hazard (2.0, 4.0)"
        ),
    )
    parser.add_argument("--no-zmq", action="store_true")
    parser.add_argument("--zmq-host", default=None)
    parser.add_argument("--zmq-port", type=int, default=None)
    parser.add_argument(
        "--scene-zenoh-topic",
        default=DEFAULT_SCENE_TOPIC,
        help="Zenoh topic used to publish SceneOutput JSON",
    )
    parser.add_argument(
        "--scene-zenoh-endpoint",
        "--scene-zenoh",
        dest="scene_zenoh_endpoint",
        default=None,
        help="Scene publisher endpoint; defaults to --zenoh-endpoint",
    )
    parser.add_argument(
        "--scene-zenoh-config",
        default=None,
        help="Scene publisher config; defaults to --zenoh-config",
    )
    parser.add_argument(
        "--no-scene-zenoh",
        action="store_true",
        help="Disable SceneOutput publishing over Zenoh",
    )
    return parser


def resolve_server_worker_path(path: str | Path) -> Path:
    """설정에 있는 상대 경로를 server_worker 디렉터리 기준으로 해석한다."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = SERVER_WORKER_DIR / resolved
    return resolved.resolve()


def load_pose_model(
    cfg: Any,
    checkpoint_path: str,
    transform,
    cams,
    device: torch.device,
    tensorrt: bool = False,
) -> torch.nn.Module:
    logger.info(
        "Loading pose model: checkpoint=%s, tensorrt=%s",
        checkpoint_path,
        tensorrt,
    )
    model = get_multi_person_pose_net(cfg, transform, cams, inference_mode="posenet")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
    model = model.eval().to(device)
    if tensorrt:
        from apps.server_worker.src.utils.tensorrt import load_tensorrt_model

        model = load_tensorrt_model(model, checkpoint_path, cfg, mode="fp16")
    logger.info("Pose model is ready: cameras=%d", len(cams))
    return model


def latest_reid_observations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int, int], dict[str, Any]] = {}
    for item in results:
        key = (
            str(item["edge_id"]),
            int(item["cam"]),
            int(item["global_id"]),
        )
        if key not in latest or int(item["frame"]) > int(latest[key]["frame"]):
            latest[key] = item
    return list(latest.values())


def associate_global_ids_to_roots(
    observations: list[dict[str, Any]],
    grid_centers: torch.Tensor,
    edge_ids: list[str],
    calibration: EdgeMetadataLoader,
) -> list[RootCandidate]:
    """각 Edge 안에서 Global ID와 root를 reprojection cost로 1:1 연결한다.

    Global ID별 여러 카메라 bbox 중심과 각 valid root의 투영점 사이 평균
    거리를 cost로 사용한다. Edge별 Hungarian assignment를 수행하므로 하나의
    root가 여러 Global ID에 중복 할당되지 않는다.
    """
    candidates: list[RootCandidate] = []

    for edge_index, edge_id in enumerate(edge_ids):
        roots = grid_centers[edge_index].detach().cpu()
        valid_root_indices = [
            index
            for index in range(len(roots))
            if roots[index, 3] >= 0 and math.isfinite(float(roots[index, 4]))
        ]
        if not valid_root_indices:
            continue

        edge_observations = [
            item for item in observations if str(item["edge_id"]) == edge_id
        ]
        global_ids = sorted({int(item["global_id"]) for item in edge_observations})
        if not global_ids:
            continue

        cameras_by_id = {
            int(camera["id"]): camera for camera in calibration[edge_id].cams
        }
        valid_roots = roots[valid_root_indices]
        projected_by_camera: dict[int, torch.Tensor] = {}
        for camera_id in {int(item["cam"]) for item in edge_observations}:
            camera = cameras_by_id.get(camera_id)
            if camera is not None:
                projected_by_camera[camera_id] = cameras.project_pose(
                    valid_roots[:, :3],
                    camera,
                )

        cost_rows = []
        matched_global_ids = []
        observation_counts = []
        for global_id in global_ids:
            distance_per_observation = []
            for item in edge_observations:
                if int(item["global_id"]) != global_id:
                    continue
                projected = projected_by_camera.get(int(item["cam"]))
                if projected is None:
                    continue

                x1, y1, x2, y2 = item["bbox"]
                center = projected.new_tensor([(x1 + x2) / 2, (y1 + y2) / 2])
                distance_per_observation.append(
                    torch.linalg.vector_norm(projected - center, dim=1)
                )

            if not distance_per_observation:
                continue
            cost_rows.append(torch.stack(distance_per_observation).mean(dim=0))
            matched_global_ids.append(global_id)
            observation_counts.append(len(distance_per_observation))

        if not cost_rows:
            continue

        cost_matrix = torch.stack(cost_rows).numpy().astype(np.float64)
        finite_costs = np.isfinite(cost_matrix) & (
            cost_matrix <= MAX_ROOT_REPROJECTION_ERROR_PX
        )
        if not finite_costs.any():
            continue
        finite_values = cost_matrix[finite_costs]
        invalid_cost = max(float(finite_values.max()) * 10 + 1, 1e9)
        assignment_cost = np.where(finite_costs, cost_matrix, invalid_cost)
        row_indices, column_indices = linear_sum_assignment(assignment_cost)

        for row_index, column_index in zip(row_indices, column_indices):
            if not finite_costs[row_index, column_index]:
                continue
            root_index = valid_root_indices[column_index]
            candidates.append(
                RootCandidate(
                    global_id=matched_global_ids[row_index],
                    edge_id=edge_id,
                    edge_index=edge_index,
                    root_index=root_index,
                    confidence=float(roots[root_index, 4]),
                    reprojection_error=float(cost_matrix[row_index, column_index]),
                    observation_count=observation_counts[row_index],
                )
            )

    return candidates


def select_highest_confidence_roots(
    candidates: list[RootCandidate],
    num_edges: int,
    num_roots: int,
) -> list[list[int | None]]:
    """Global ID마다 confidence가 가장 높은 Edge/root 후보만 남긴다."""
    winners: dict[int, RootCandidate] = {}
    for candidate in candidates:
        current = winners.get(candidate.global_id)
        candidate_rank = (
            candidate.confidence,
            -candidate.reprojection_error,
            candidate.observation_count,
            -candidate.edge_index,
            -candidate.root_index,
        )
        if current is None:
            winners[candidate.global_id] = candidate
            continue
        current_rank = (
            current.confidence,
            -current.reprojection_error,
            current.observation_count,
            -current.edge_index,
            -current.root_index,
        )
        if candidate_rank > current_rank:
            winners[candidate.global_id] = candidate

    ids = [[None] * num_roots for _ in range(num_edges)]
    for global_id, winner in winners.items():
        ids[winner.edge_index][winner.root_index] = global_id
    return ids


def assign_global_ids(
    reid_results: list[dict[str, Any]],
    grid_centers: torch.Tensor,
    edge_ids: list[str],
    calibration: EdgeMetadataLoader,
) -> list[list[int | None]]:
    """Re-ID Global ID를 confidence가 가장 높은 Edge/root에 매핑한다.

    먼저 각 Edge 안에서 Global ID와 root를 reprojection cost로 1:1
    association한다. 동일 Global ID가 여러 Edge에 존재하면 root confidence
    (5차원 벡터의 마지막 값)가 가장 높은 후보만 남긴다.
    """
    if grid_centers.ndim != 3 or grid_centers.shape[-1] != 5:
        raise ValueError("grid_centers must have shape [num_edges, num_roots, 5]")
    if grid_centers.shape[0] != len(edge_ids):
        raise ValueError("grid_centers batch size must match the number of edge_ids")

    if not reid_results:
        return [[None] * grid_centers.shape[1] for _ in edge_ids]

    observations = latest_reid_observations(reid_results)
    candidates = associate_global_ids_to_roots(
        observations,
        grid_centers,
        edge_ids,
        calibration,
    )
    return select_highest_confidence_roots(
        candidates,
        num_edges=len(edge_ids),
        num_roots=grid_centers.shape[1],
    )


def assign_all_lod2(global_ids: list[int]) -> LodAssignments:
    """현재 기본 정책: 매핑된 모든 global ID를 LOD 2로 배정한다."""
    return {global_id: 2 for global_id in global_ids}


def lod_assign(
    ids: list[list[int | None]],
    callback: LodAssignCallback,
) -> LodAssignments:
    """root에 매핑된 global ID를 callback 정책으로 LOD에 배정한다.

    callback은 정렬된 고유 global ID 목록을 받고, 각 ID에 대한 LOD
    0, 1, 2를 모두 포함하는 mapping을 반환해야 한다.
    """
    global_ids = sorted(
        {
            global_id
            for edge_ids in ids
            for global_id in edge_ids
            if global_id is not None
        }
    )
    assignments = callback(global_ids)

    if set(assignments) != set(global_ids):
        raise ValueError("LOD callback must assign every global ID exactly once")
    invalid_assignments = {
        global_id: lod for global_id, lod in assignments.items() if lod not in (0, 1, 2)
    }
    if invalid_assignments:
        raise ValueError(f"LOD values must be 0, 1, or 2: {invalid_assignments}")

    return assignments


def build_priority_input(
    ids: list[list[int | None]],
    roots: torch.Tensor,
) -> dict[str, list[dict[str, Any]]]:
    """Global ID와 root를 PriorityEngine의 metre 단위 입력으로 변환한다."""
    if roots.ndim != 3 or roots.shape[-1] != 5:
        raise ValueError("roots must have shape [num_edges, num_roots, 5]")
    if len(ids) != roots.shape[0] or any(
        len(edge_ids) != roots.shape[1] for edge_ids in ids
    ):
        raise ValueError("ids shape must match the Edge/root dimensions")

    roots_cpu = roots.detach().cpu()
    persons = []
    seen_global_ids = set()
    for edge_index, edge_ids in enumerate(ids):
        for root_index, global_id in enumerate(edge_ids):
            if global_id is None:
                continue
            if global_id in seen_global_ids:
                raise ValueError(f"duplicate global ID in roots: {global_id}")
            seen_global_ids.add(global_id)

            raw_position = roots_cpu[edge_index, root_index, :3]
            if not torch.isfinite(raw_position).all():
                raise ValueError(f"root position must be finite: global_id={global_id}")
            x, y, z = (
                float(value) * ROOT_POSITION_TO_METERS
                for value in raw_position.tolist()
            )
            persons.append(
                {
                    "track_id": global_id,
                    "position": {"x": x, "y": y, "z": z},
                }
            )

    return {"persons": persons}


def assign_priority_lods(
    ids: list[list[int | None]],
    roots: torch.Tensor,
    timestamps: list[float],
    engine: PriorityEngine,
    lod2_count: int,
) -> LodAssignments:
    """PriorityEngine rank 상위 인원을 LOD 2, 나머지를 LOD 1로 배정한다."""
    if isinstance(lod2_count, bool) or not isinstance(lod2_count, int):
        raise TypeError("lod2_count must be an integer")
    if lod2_count < 0:
        raise ValueError("lod2_count must not be negative")
    if len(timestamps) != roots.shape[0] or not timestamps:
        raise ValueError("timestamps must contain one value per Edge")

    timestamp_values = [float(timestamp) for timestamp in timestamps]
    if not all(math.isfinite(timestamp) for timestamp in timestamp_values):
        raise ValueError("timestamps must be finite")
    scene_timestamp = sum(timestamp_values) / len(timestamp_values)
    ranked_result = engine.assign_priority(
        build_priority_input(ids, roots),
        timestamp=scene_timestamp,
        selected_count=lod2_count,
    )
    assignments = {
        int(person["track_id"]): 2 if int(person["rank"]) <= lod2_count else 1
        for person in ranked_result["persons"]
    }

    # 기존 lod_assign의 ID 완전성 및 LOD 범위 검증을 그대로 적용한다.
    return lod_assign(ids, lambda _global_ids: assignments)


def select_lod2_roots(
    roots: torch.Tensor,
    ids: list[list[int | None]],
    lod_by_id: LodAssignments,
) -> torch.Tensor:
    """LOD 2 ID가 매핑된 root 슬롯만 pose 추론 대상으로 남긴다."""
    if roots.ndim != 3 or roots.shape[-1] != 5:
        raise ValueError("roots must have shape [num_edges, num_roots, 5]")
    if len(ids) != roots.shape[0] or any(
        len(edge_ids) != roots.shape[1] for edge_ids in ids
    ):
        raise ValueError("ids shape must match the Edge/root dimensions")

    mapped_ids = {
        global_id for edge_ids in ids for global_id in edge_ids if global_id is not None
    }
    missing_ids = mapped_ids - set(lod_by_id)
    if missing_ids:
        raise ValueError(
            f"LOD assignments are missing global IDs: {sorted(missing_ids)}"
        )

    lod2_mask = torch.tensor(
        [
            [
                global_id is not None and lod_by_id[global_id] == 2
                for global_id in edge_ids
            ]
            for edge_ids in ids
        ],
        dtype=torch.bool,
        device=roots.device,
    )
    selected_roots = roots.clone()
    selected_roots[:, :, 3].masked_fill_(~lod2_mask, -1)
    return selected_roots


def run_pose_models(
    pose_models: dict[str, torch.nn.Module],
    heatmaps: list[torch.Tensor],
    roots: torch.Tensor,
    edge_ids: list[str],
    ids: list[list[int | None]],
    lod_by_id: LodAssignments,
) -> dict[str, torch.Tensor]:
    """Edge별 모델에 해당 heatmap과 LOD 2 roots를 넣어 실행한다."""
    if not heatmaps:
        raise ValueError("heatmaps must contain at least one camera view")
    if roots.ndim != 3 or roots.shape[0] != len(edge_ids):
        raise ValueError("roots batch size must match the number of edge_ids")
    if any(view.shape[0] != len(edge_ids) for view in heatmaps):
        raise ValueError("every heatmap view must contain every Edge")

    lod2_roots = select_lod2_roots(roots, ids, lod_by_id)
    predictions = []
    result_roots = []
    for edge_index, edge_id in enumerate(edge_ids):
        model = pose_models[edge_id]
        edge_heatmaps = [view[edge_index : edge_index + 1] for view in heatmaps]
        edge_roots = lod2_roots[edge_index : edge_index + 1]
        validate_pose_model_inputs(
            model,
            edge_id,
            edge_heatmaps,
            edge_roots,
        )
        edge_result = model(
            input_heatmaps=edge_heatmaps,
            grid_centers=edge_roots,
        )
        predictions.append(edge_result["pred"])
        result_roots.append(edge_result["grid_centers"])

    return {
        "pred": torch.cat(predictions, dim=0),
        "grid_centers": torch.cat(result_roots, dim=0),
    }


def build_scene_output(
    pose_result: dict[str, torch.Tensor],
    roots: torch.Tensor,
    ids: list[list[int | None]],
    lod_by_id: LodAssignments,
    timestamps: list[float],
    edge_ids: list[str],
    sync_spread: float,
    max_people: int = DEFAULT_MAX_OUTPUT_PEOPLE,
) -> SceneOutput:
    """Tensor 결과를 다른 프로세스가 소비할 수 있는 사람 entity로 변환한다."""
    if max_people < 1:
        raise ValueError("max_people must be at least 1")
    if roots.ndim != 3 or roots.shape[-1] != 5:
        raise ValueError("roots must have shape [num_edges, num_roots, 5]")
    if roots.shape[0] != len(edge_ids):
        raise ValueError("roots batch size must match edge_ids")
    if len(ids) != len(edge_ids) or any(
        len(edge_root_ids) != roots.shape[1] for edge_root_ids in ids
    ):
        raise ValueError("ids shape must match roots")
    if len(timestamps) != len(edge_ids):
        raise ValueError("timestamps must contain one value per Edge")

    predictions = pose_result["pred"]
    result_roots = pose_result["grid_centers"]
    expected_prediction_prefix = (roots.shape[0], roots.shape[1])
    if tuple(predictions.shape[:2]) != expected_prediction_prefix:
        raise ValueError("pose predictions must match roots")
    if tuple(result_roots.shape[:2]) != expected_prediction_prefix:
        raise ValueError("pose result roots must match roots")

    roots_cpu = roots.detach().cpu()
    predictions_cpu = predictions.detach().cpu()
    result_roots_cpu = result_roots.detach().cpu()
    people = []
    for edge_index, edge_id in enumerate(edge_ids):
        for root_index, global_id in enumerate(ids[edge_index]):
            if global_id is None:
                continue

            lod = lod_by_id[global_id]
            raw_root = roots_cpu[edge_index, root_index]
            root_output = RootOutput(
                edge_id=edge_id,
                candidate_index=root_index,
                position=[float(value) for value in raw_root[:3].tolist()],
                confidence=float(raw_root[4]),
                timestamp=float(timestamps[edge_index]),
            )

            pose_output = None
            pose_is_valid = bool(result_roots_cpu[edge_index, root_index, 3] >= 0)
            if lod == 2 and pose_is_valid:
                joint_positions = predictions_cpu[edge_index, root_index, :, :3]
                if torch.isfinite(joint_positions).all() and torch.any(
                    joint_positions != 0
                ):
                    pose_output = PoseOutput(
                        joint_format=(f"voxelpose_{joint_positions.shape[0]}j_xyz"),
                        joints=[
                            [float(value) for value in joint]
                            for joint in joint_positions.tolist()
                        ],
                    )

            people.append(
                PersonEntity(
                    global_id=global_id,
                    lod=lod,
                    root=root_output,
                    pose=pose_output,
                )
            )

    people.sort(
        key=lambda person: (
            -person.lod,
            -person.root.confidence,
            person.global_id,
        )
    )
    if len(people) > max_people:
        logger.warning(
            "Scene contains %d people; output is limited to %d",
            len(people),
            max_people,
        )
        people = people[:max_people]

    scene_timestamp = sum(float(timestamp) for timestamp in timestamps) / len(
        timestamps
    )
    return SceneOutput(
        timestamp=scene_timestamp,
        sync_spread_seconds=float(sync_spread),
        people=people,
    )


def validate_pose_model_inputs(
    model: torch.nn.Module,
    edge_id: str,
    heatmaps: list[torch.Tensor],
    roots: torch.Tensor,
) -> None:
    """MultiPersonPoseNetSSV가 요구하는 입력 계약을 검증한다."""
    expected_views = len(model.pose_net.project_layer.cams)
    expected_joints = int(model.num_joints)
    expected_candidates = int(model.num_cand)
    heatmap_width, heatmap_height = (
        int(value) for value in model.pose_net.project_layer.heatmap_size
    )
    expected_heatmap_shape = (
        1,
        expected_joints,
        heatmap_height,
        heatmap_width,
    )
    expected_root_shape = (1, expected_candidates, 5)

    if len(heatmaps) != expected_views:
        raise ValueError(
            f"{edge_id}: expected {expected_views} heatmap views, got {len(heatmaps)}"
        )
    for camera_index, heatmap in enumerate(heatmaps):
        if tuple(heatmap.shape) != expected_heatmap_shape:
            raise ValueError(
                f"{edge_id}/camera-{camera_index + 1}: expected heatmap "
                f"shape {expected_heatmap_shape}, got {tuple(heatmap.shape)}"
            )
        if heatmap.dtype != torch.float32:
            raise ValueError(
                f"{edge_id}/camera-{camera_index + 1}: heatmap must be float32"
            )

    if tuple(roots.shape) != expected_root_shape:
        raise ValueError(
            f"{edge_id}: expected roots shape {expected_root_shape}, "
            f"got {tuple(roots.shape)}"
        )
    if roots.dtype != torch.float32:
        raise ValueError(f"{edge_id}: roots must be float32")
    if any(heatmap.device != roots.device for heatmap in heatmaps):
        raise ValueError(f"{edge_id}: heatmaps and roots must use one device")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = get_parser().parse_args()
    logger.info("Starting server inference: args=%s", args)
    if args.max_output_people < 1:
        raise ValueError("--max-output-people must be at least 1")
    if args.lod2_count < 0:
        raise ValueError("--lod2-count must not be negative")

    if args.cfg_focus:
        logger.info("Loading focus config: %s", args.cfg_focus)
        update_focus_config(args.cfg_focus)
    pose_config_path = (
        Path(args.pose_config).expanduser().resolve()
        if args.pose_config
        else resolve_server_worker_path(focus_config.POSENET.CONFIG)
    )
    update_sp3d_config(str(pose_config_path))
    edge_metadata_path = Path(args.deployment).expanduser().resolve()
    spatial_context = load_spatial_context(edge_metadata_path)
    sp3d_config.SPATIAL_CONTEXT = spatial_context
    logger.info("Pose config is ready: %s", pose_config_path)

    device = setup_cuda()
    use_tensorrt = (
        bool(focus_config.POSENET.TENSORRT) if args.tensorrt is None else args.tensorrt
    )
    checkpoint_path = resolve_server_worker_path(focus_config.POSENET.CKPT)
    zmq_host = args.zmq_host or focus_config.ZMQ_SERVER
    zmq_port = (
        args.zmq_port if args.zmq_port is not None else int(focus_config.ZMQ_PORT)
    )
    logger.info(
        "Runtime options: device=%s, tensorrt=%s, zmq=%s:%s",
        device,
        use_tensorrt,
        zmq_host,
        zmq_port,
    )

    logger.info(
        "[2/8] Loading Edge metadata: path=%s edge_ids=%s",
        edge_metadata_path,
        args.edge_ids or "all enabled",
    )
    edge_metadata = EdgeMetadataLoader(
        sp3d_config,
        args.edge_ids,
        edge_metadata_path,
    )
    edge_ids = edge_metadata.edge_ids
    logger.info("Edge metadata is ready: topics=%s", edge_metadata.topics)
    expected_workspace_ids = None
    if spatial_context is not None:
        expected_workspace_ids = {}
        for edge_id in edge_ids:
            workspace = build_edge_workspace(
                spatial_context,
                edge_id,
                edge_metadata[edge_id].cams,
                sp3d_config.NETWORK.IMAGE_SIZE_ORIG,
                sp3d_config.MULTI_PERSON.SPACE_SIZE,
                sp3d_config.MULTI_PERSON.INITIAL_CUBE_SIZE,
            )
            expected_workspace_ids[edge_id] = workspace.workspace_id
            logger.info(
                "Workspace contract is ready: edge=%s source=%s cube=%s id=%s",
                edge_id,
                workspace.source,
                workspace.cube_size.tolist(),
                workspace.workspace_id[:12],
            )
    # Re-ID model
    """
    먼저 Re-ID를 수행해서 global-id를 구하고
    겹치는 ID가 있으면, root 신뢰도가 높은 것을 살려야함
    그다음 LoD 기준에 따라서 Pose 대상을 선정
    """
    reid = ClusteringSliding(edges=edge_ids, window_size=10)
    logger.info("[3/8] Re-ID clustering is ready: edges=%s", edge_ids)
    # LoD assignment
    priority_hazards = (
        tuple((float(x), float(y)) for x, y in args.priority_hazard)
        if args.priority_hazard
        else PriorityConfig().hazards
    )
    priority_engine = PriorityEngine(PriorityConfig(hazards=priority_hazards))
    logger.info(
        "Priority-based LOD assignment is ready: lod2_count=%d hazards=%s",
        args.lod2_count,
        priority_hazards,
    )
    # pose_model
    """
    Edge마다 할당되면 개별 pose_model이 선언됨
    LoD=2 선별된 ID들이 각 pose_model에서 추론됨
    """
    pose_models = {}
    for edge_id in edge_ids:
        metadata = edge_metadata[edge_id]
        pose_models[edge_id] = load_pose_model(
            sp3d_config,
            str(checkpoint_path),
            metadata.transform,
            metadata.cams,
            device,
            use_tensorrt,
        )
    logger.info("[4/8] Pose models are ready: count=%d", len(pose_models))

    # 제노 로더
    logger.info("[5/8] Connecting to Zenoh: endpoint=%s", args.zenoh_endpoint)
    loader = ZenohDataLoader(
        edge_ids=edge_ids,
        device=device,
        topics=edge_metadata.topics,
        expected_camera_ids={
            edge_id: [int(camera["id"]) for camera in edge_metadata[edge_id].cams]
            for edge_id in edge_ids
        },
        expected_spatial_context=(
            spatial_context.identity() if spatial_context is not None else None
        ),
        expected_workspace_ids=expected_workspace_ids,
        endpoint=args.zenoh_endpoint,
        config_path=args.zenoh_config,
        buffer_size=args.buffer_size,
        sync_tolerance=args.sync_tolerance,
    )
    logger.info("Zenoh data loader is ready; waiting for synchronized batches")
    zmq_output = None
    scene_publisher = None
    try:
        zmq_output = (
            Protocol(zmq_host, zmq_port)
            if ENABLE_ZMQ_OUTPUT and not args.no_zmq
            else None
        )
        if zmq_output is None:
            logger.info("[6/8] ZMQ scene output is disabled")
        else:
            logger.info("[6/8] ZMQ scene output is ready")

        if not args.no_scene_zenoh:
            scene_publisher = ZenohScenePublisher(
                topic=args.scene_zenoh_topic,
                endpoint=args.scene_zenoh_endpoint or args.zenoh_endpoint,
                config_path=args.scene_zenoh_config or args.zenoh_config,
            )
            logger.info(
                "[6/8] Zenoh scene output is ready: topic=%s",
                args.scene_zenoh_topic,
            )
        else:
            logger.info("[6/8] Zenoh scene output is disabled")
    except Exception:
        loader.close()
        if zmq_output is not None:
            zmq_output.close()
        if scene_publisher is not None:
            scene_publisher.close()
        raise

    try:
        logger.info("[7/8] Entering inference loop")
        batch_count = 0
        last_wait_log = time.monotonic()
        with torch.inference_mode():
            while True:
                prepared = loader.get(timeout=args.receive_timeout)
                if prepared is None:
                    now = time.monotonic()
                    if now - last_wait_log >= 10:
                        logger.info(
                            "Waiting for synchronized Zenoh input: dropped=%s",
                            loader.subscriber.dropped,
                        )
                        last_wait_log = now
                    continue

                batch_count += 1
                if batch_count == 1:
                    logger.info("First synchronized batch received")
                started_at = time.perf_counter()
                try:
                    heatmaps, roots, reid_items, timestamps, sync_spread = prepared
                    reid_results = reid.process_realtime(reid_items) or []
                    ids = assign_global_ids(
                        reid_results,
                        roots,
                        edge_ids,
                        edge_metadata,
                    )

                    lod_by_id = assign_priority_lods(
                        ids,
                        roots,
                        timestamps,
                        priority_engine,
                        args.lod2_count,
                    )
                    logger.debug("LOD assignments: %s", lod_by_id)

                    pose_result = run_pose_models(
                        pose_models,
                        heatmaps,
                        roots,
                        edge_ids,
                        ids,
                        lod_by_id,
                    )
                    scene_output = build_scene_output(
                        pose_result,
                        roots,
                        ids,
                        lod_by_id,
                        timestamps,
                        edge_ids,
                        sync_spread,
                        max_people=args.max_output_people,
                    )
                    logger.info(
                        "Scene output is ready: people=%d",
                        len(scene_output.people),
                    )
                    # ===== TEMP VISER DEBUG: delete with the block near constants =====
                    if ENABLE_VISER_DEBUG_OUTPUT:
                        save_scene_for_viser(scene_output)
                    # ===== END TEMP VISER DEBUG =====

                    scene_payload = scene_output.to_dict()
                    if zmq_output is not None and not zmq_output.send_scene(
                        scene_payload
                    ):
                        logger.error("Failed to send scene output over ZMQ")
                    if (
                        scene_publisher is not None
                        and not scene_publisher.send_scene(scene_payload)
                    ):
                        logger.error("Failed to send scene output over Zenoh")
                except Exception:
                    logger.exception("Failed to process synchronized batch")
                    continue

                elapsed = (time.perf_counter() - started_at) * 1000
                logger.info(
                    "Batch %d complete: inference=%.1fms sync_spread=%.1fms dropped=%s",
                    batch_count,
                    elapsed,
                    sync_spread * 1000,
                    loader.subscriber.dropped,
                )
    except KeyboardInterrupt:
        logger.info("Stopping server inference")
    finally:
        logger.info("[8/8] Closing inference resources")
        loader.close()
        if zmq_output is not None:
            zmq_output.close()
        if scene_publisher is not None:
            scene_publisher.close()


if __name__ == "__main__":
    main()
