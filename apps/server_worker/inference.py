# ruff: noqa: E402

import argparse
from collections import Counter
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
    encode_scene,
)
from apps.server_worker.src.protocol.zenoh import ZenohDataLoader
from apps.server_worker.src.protocol.zmq import Protocol
from apps.server_worker.src.reid.sliding_clustering import ClusteringSliding
from apps.server_worker.src.utils.edgemetadata import EdgeMetadataLoader
from apps.server_worker.src.utils.edgemetadata import DEFAULT_METADATA_PATH
from apps.server_worker.src.utils.telemetry import (
    ResourceSampler,
    TelemetryRecorder,
    collect_runtime_metadata,
    tensor_nbytes,
)


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

ENABLE_ZMQ_OUTPUT = False
VISER_DEBUG_OUTPUT_PATH = SERVER_WORKER_DIR / "data" / "viser_scenes.jsonl"
DEFAULT_METRICS_DIR = SERVER_WORKER_DIR / "data" / "metrics"
DEFAULT_METRICS_SAMPLE_EVERY = 30
DEFAULT_STATUS_LOG_EVERY = 30


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
        "--metrics-dir",
        default=str(DEFAULT_METRICS_DIR),
        help="Root directory for per-run JSONL telemetry",
    )
    parser.add_argument(
        "--metrics-run-label",
        help="Human-readable experiment label stored with the run",
    )
    parser.add_argument(
        "--metrics-tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Experiment tag; repeat for load level, dataset, scenario, etc.",
    )
    parser.add_argument(
        "--metrics-resource-interval",
        type=float,
        default=5.0,
        help="Seconds between CPU/RAM/GPU/network samples; 0 samples every batch",
    )
    parser.add_argument(
        "--metrics-sample-every",
        type=int,
        default=DEFAULT_METRICS_SAMPLE_EVERY,
        metavar="N",
        help="Persist detailed telemetry for the first and every Nth batch",
    )
    parser.add_argument(
        "--metrics-include-reid-observations",
        action="store_true",
        help="Include per-detection bbox/ID metadata in sampled batches",
    )
    parser.add_argument(
        "--metrics-include-gpu-device-stats",
        action="store_true",
        help="Query optional NVML utilization/power/temperature counters",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable structured inference telemetry",
    )
    parser.add_argument(
        "--status-log-every",
        type=int,
        default=DEFAULT_STATUS_LOG_EVERY,
        metavar="N",
        help="Write the batch status log for the first and every Nth batch",
    )
    parser.add_argument(
        "--viser-debug-output",
        nargs="?",
        const=str(VISER_DEBUG_OUTPUT_PATH),
        default=None,
        metavar="PATH",
        help="Enable synchronous Viser JSONL debug output (disabled by default)",
    )
    parser.add_argument(
        "--max-output-people",
        type=int,
        default=DEFAULT_MAX_OUTPUT_PEOPLE,
        help="Maximum number of person entities emitted per synchronized scene",
    )
    lod_policy = parser.add_mutually_exclusive_group()
    lod_policy.add_argument(
        "--lod2-count",
        type=int,
        default=DEFAULT_LOD2_PEOPLE,
        help="Number of highest-priority people assigned to LOD 2",
    )
    lod_policy.add_argument(
        "--all-lod2",
        action="store_true",
        help="Assign every mapped global ID to LOD 2",
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


def parse_metrics_tags(values: list[str]) -> dict[str, str]:
    """Parse repeatable KEY=VALUE experiment labels."""
    tags = {}
    for value in values:
        key, separator, tag_value = value.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Invalid --metrics-tag {value!r}; expected KEY=VALUE")
        tags[key] = tag_value.strip()
    return tags


def count_reid_items(
    items: list[dict[str, Any]],
    edge_ids: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Count Re-ID observations without persisting feature vectors or bboxes."""
    by_edge = Counter(str(item.get("edge_id", "unknown")) for item in items)
    by_camera = Counter(
        f"{item.get('edge_id', 'unknown')}:{item.get('cam', 'unknown')}"
        for item in items
    )
    return (
        {edge_id: int(by_edge.get(edge_id, 0)) for edge_id in edge_ids},
        dict(sorted(by_camera.items())),
    )


def estimate_reid_feature_bytes(items: list[dict[str, Any]]) -> int:
    total = 0
    for item in items:
        feature = item.get("feature")
        if feature is None:
            continue
        if isinstance(feature, np.ndarray):
            total += int(feature.nbytes)
        elif isinstance(feature, torch.Tensor):
            total += tensor_nbytes(feature)
        else:
            total += int(np.asarray(feature, dtype=np.float32).nbytes)
    return total


def reid_observation_key(item: dict[str, Any]) -> tuple[Any, ...]:
    bbox = tuple(float(value) for value in item.get("bbox") or [])
    return (
        str(item.get("edge_id", "unknown")),
        item.get("cam"),
        item.get("frame"),
        item.get("person_idx"),
        bbox,
    )


def reid_observation_metadata(
    reid_items: list[dict[str, Any]],
    reid_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep evaluation identifiers and boxes while excluding embeddings."""
    assigned_ids = {
        reid_observation_key(item): int(item["global_id"])
        for item in reid_results
    }
    observations = []
    for item in reid_items:
        observation = {
            "edge_id": str(item.get("edge_id", "unknown")),
            "camera_id": (
                None if item.get("cam") is None else int(item["cam"])
            ),
            "frame": (
                None if item.get("frame") is None else int(item["frame"])
            ),
            "person_index": (
                None
                if item.get("person_idx") is None
                else int(item["person_idx"])
            ),
            "bbox_xyxy": [
                float(value) for value in item.get("bbox") or []
            ],
            "global_id": assigned_ids.get(reid_observation_key(item)),
        }
        observations.append(observation)
    return observations


def build_reid_metrics(
    reid_items: list[dict[str, Any]],
    reid_results: list[dict[str, Any]],
    mapped_ids: set[int],
    previous_mapped_ids: set[int],
    seen_global_ids: set[int],
    tracker: ClusteringSliding,
    *,
    include_observations: bool = False,
) -> dict[str, Any]:
    """Build online identity-continuity indicators (not ground-truth accuracy)."""
    result_ids = {int(item["global_id"]) for item in reid_results}
    entered_ids = mapped_ids - previous_mapped_ids
    exited_ids = previous_mapped_ids - mapped_ids
    union_ids = mapped_ids | previous_mapped_ids
    edges_by_global_id: dict[int, set[str]] = {}
    for item in reid_results:
        edges_by_global_id.setdefault(int(item["global_id"]), set()).add(
            str(item["edge_id"])
        )

    metrics = {
        "input_observations": len(reid_items),
        "confirmed_observations": len(reid_results),
        "confirmed_global_ids": sorted(result_ids),
        "mapped_global_ids": sorted(mapped_ids),
        "new_global_ids_this_run": sorted(result_ids - seen_global_ids),
        "entered_since_previous_batch": sorted(entered_ids),
        "exited_since_previous_batch": sorted(exited_ids),
        "retained_from_previous_batch": sorted(
            mapped_ids & previous_mapped_ids
        ),
        "identity_churn_ratio": (
            (len(entered_ids) + len(exited_ids)) / len(union_ids)
            if union_ids
            else 0.0
        ),
        "observation_confirmation_ratio": (
            len(reid_results) / len(reid_items) if reid_items else None
        ),
        "root_association_ratio": (
            len(mapped_ids) / len(result_ids) if result_ids else None
        ),
        "global_ids_seen_on_multiple_edges": sorted(
            global_id
            for global_id, observed_edges in edges_by_global_id.items()
            if len(observed_edges) > 1
        ),
        "tracker": tracker.metrics_snapshot(),
    }
    if include_observations:
        metrics["observations"] = reid_observation_metadata(
            reid_items,
            reid_results,
        )
    return metrics


def unix_input_age_ms(timestamps: list[float], now: float) -> float | None:
    """Return source-to-output age only when timestamps look like Unix time."""
    if not timestamps:
        return None
    earliest = min(float(timestamp) for timestamp in timestamps)
    if earliest < 946684800 or earliest > now + 60:
        return None
    # Keep small negative values visible: they reveal Edge/server clock skew.
    return (now - earliest) * 1000.0


def counter_deltas(
    current: dict[str, int],
    previous: dict[str, int],
) -> dict[str, int]:
    return {
        key: max(0, int(value) - int(previous.get(key, 0)))
        for key, value in current.items()
    }


def queue_metrics(
    loader: ZenohDataLoader,
    scene_publisher: ZenohScenePublisher | None,
    telemetry: TelemetryRecorder | None,
) -> dict[str, Any]:
    return {
        "input_subscriber_depth_by_edge": {
            edge_id: input_queue.qsize()
            for edge_id, input_queue in loader.subscriber.queues.items()
        },
        "synchronization_buffer_depth_by_edge": {
            edge_id: len(buffer) for edge_id, buffer in loader.buffers.items()
        },
        "scene_output_depth": (
            None if scene_publisher is None else scene_publisher.queue.qsize()
        ),
        "telemetry_writer_depth": (
            None if telemetry is None else telemetry.queue_depth
        ),
    }


def transport_counter_metrics(
    loader: ZenohDataLoader,
    scene_publisher: ZenohScenePublisher | None,
    previous: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Snapshot application-level Zenoh traffic and calculate record deltas."""
    subscriber = loader.subscriber
    input_messages = {
        edge_id: int(value)
        for edge_id, value in subscriber.received_messages.items()
    }
    input_bytes = {
        edge_id: int(value) for edge_id, value in subscriber.received_bytes.items()
    }
    invalid_messages = {
        edge_id: int(value)
        for edge_id, value in subscriber.invalid_messages.items()
    }
    output_messages = (
        0 if scene_publisher is None else int(scene_publisher.sent_messages)
    )
    output_bytes = 0 if scene_publisher is None else int(scene_publisher.sent_bytes)
    synchronization_discarded = {
        edge_id: int(value)
        for edge_id, value in loader.synchronization_discarded.items()
    }
    current = {
        "input_messages": input_messages,
        "input_bytes": input_bytes,
        "invalid_messages": invalid_messages,
        "output_messages": output_messages,
        "output_bytes": output_bytes,
        "synchronization_discarded": synchronization_discarded,
        "decode_failures": int(loader.decode_failures),
        "prepare_failures": int(loader.prepare_failures),
    }
    metrics = {
        "input_received_messages_total_by_edge": input_messages,
        "input_received_messages_delta_by_edge": counter_deltas(
            input_messages,
            previous.get("input_messages", {}),
        ),
        "input_received_bytes_total_by_edge": input_bytes,
        "input_received_bytes_delta_by_edge": counter_deltas(
            input_bytes,
            previous.get("input_bytes", {}),
        ),
        "input_invalid_messages_total_by_edge": invalid_messages,
        "input_invalid_messages_delta_by_edge": counter_deltas(
            invalid_messages,
            previous.get("invalid_messages", {}),
        ),
        "synchronization_discarded_total_by_edge": synchronization_discarded,
        "synchronization_discarded_delta_by_edge": counter_deltas(
            synchronization_discarded,
            previous.get("synchronization_discarded", {}),
        ),
        "decode_failures_total": int(loader.decode_failures),
        "decode_failures_delta": max(
            0,
            int(loader.decode_failures) - int(previous.get("decode_failures", 0)),
        ),
        "prepare_failures_total": int(loader.prepare_failures),
        "prepare_failures_delta": max(
            0,
            int(loader.prepare_failures)
            - int(previous.get("prepare_failures", 0)),
        ),
        "scene_sent_messages_total": output_messages,
        "scene_sent_messages_delta": max(
            0,
            output_messages - int(previous.get("output_messages", 0)),
        ),
        "scene_sent_bytes_total": output_bytes,
        "scene_sent_bytes_delta": max(
            0,
            output_bytes - int(previous.get("output_bytes", 0)),
        ),
        "scene_router_ids": (
            [] if scene_publisher is None else list(scene_publisher.router_ids)
        ),
        "scene_link_count": (
            None if scene_publisher is None else scene_publisher.link_count
        ),
    }
    return metrics, current


def sample_resources(
    sampler: ResourceSampler | None,
    *,
    force: bool = False,
) -> dict[str, Any] | None:
    if sampler is None:
        return None
    try:
        return sampler.sample(force=force)
    except Exception as exc:
        logger.warning("Resource telemetry sample failed: %s", exc)
        return {"sampling_error": f"{type(exc).__name__}: {exc}"}


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
    main_started_at = time.perf_counter()
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
    if args.metrics_resource_interval < 0:
        raise ValueError("--metrics-resource-interval must not be negative")
    if args.metrics_sample_every < 1:
        raise ValueError("--metrics-sample-every must be at least 1")
    if args.status_log_every < 1:
        raise ValueError("--status-log-every must be at least 1")
    metrics_tags = parse_metrics_tags(args.metrics_tag)

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
        "LOD assignment is ready: policy=%s lod2_count=%d hazards=%s",
        "all_lod2" if args.all_lod2 else "priority",
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
    camera_ids_by_edge = {
        edge_id: [int(camera["id"]) for camera in edge_metadata[edge_id].cams]
        for edge_id in edge_ids
    }
    loader = ZenohDataLoader(
        edge_ids=edge_ids,
        device=device,
        topics=edge_metadata.topics,
        expected_camera_ids=camera_ids_by_edge,
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
    telemetry = None
    resource_sampler = None
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

        if not args.no_metrics:
            run_metadata = {
                "experiment": {
                    "label": args.metrics_run_label,
                    "tags": metrics_tags,
                },
                "configuration": {
                    "arguments": vars(args),
                    "pose_config_path": str(pose_config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "deployment_path": str(edge_metadata_path),
                    "tensorrt": use_tensorrt,
                    "lod_policy": "all_lod2" if args.all_lod2 else "priority",
                    "priority_hazards_metres": priority_hazards,
                    "telemetry_sampling": {
                        "batch_interval": args.metrics_sample_every,
                        "resource_interval_seconds": (
                            args.metrics_resource_interval
                        ),
                        "include_reid_observations": (
                            args.metrics_include_reid_observations
                        ),
                        "include_gpu_device_stats": (
                            args.metrics_include_gpu_device_stats
                        ),
                    },
                },
                "topology": {
                    "edge_ids": edge_ids,
                    "camera_ids_by_edge": camera_ids_by_edge,
                    "input_topics": edge_metadata.topics,
                    "workspace_ids": expected_workspace_ids,
                    "scene_output_topic": (
                        None if args.no_scene_zenoh else args.scene_zenoh_topic
                    ),
                },
                "runtime": collect_runtime_metadata(device, PROJECT_ROOT),
            }
            try:
                telemetry = TelemetryRecorder(args.metrics_dir, run_metadata)
                resource_sampler = ResourceSampler(
                    device,
                    interval_seconds=args.metrics_resource_interval,
                    include_device_metrics=(
                        args.metrics_include_gpu_device_stats
                    ),
                )
                telemetry.record_event(
                    {
                        "event": "startup_complete",
                        "startup_ms": (
                            time.perf_counter() - main_started_at
                        )
                        * 1000.0,
                        "resource": sample_resources(
                            resource_sampler,
                            force=True,
                        ),
                    }
                )
                logger.info("Inference telemetry is ready: %s", telemetry.run_dir)
            except Exception as exc:
                if telemetry is not None:
                    telemetry.close({"initialization_error": repr(exc)})
                telemetry = None
                resource_sampler = None
                logger.exception("Failed to initialize inference telemetry")
        else:
            logger.info("Inference telemetry is disabled")
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
        last_batch_received_at = None
        previous_mapped_ids: set[int] = set()
        seen_global_ids: set[int] = set()
        previous_input_dropped = {edge_id: 0 for edge_id in edge_ids}
        previous_output_dropped = 0
        previous_transport_counters: dict[str, Any] = {}
        pose_cuda_start = (
            torch.cuda.Event(enable_timing=True) if telemetry is not None else None
        )
        pose_cuda_end = (
            torch.cuda.Event(enable_timing=True) if telemetry is not None else None
        )
        viser_debug_output_path = (
            None
            if args.viser_debug_output is None
            else Path(args.viser_debug_output).expanduser().resolve()
        )
        if viser_debug_output_path is not None:
            logger.warning(
                "Synchronous Viser debug output is enabled: %s",
                viser_debug_output_path,
            )
        with torch.inference_mode():
            while True:
                receive_started_at = time.perf_counter()
                prepared = loader.get(timeout=args.receive_timeout)
                batch_received_at = time.perf_counter()
                if prepared is None:
                    now = time.monotonic()
                    if now - last_wait_log >= 10:
                        input_dropped = {
                            edge_id: int(value)
                            for edge_id, value in loader.subscriber.dropped.items()
                        }
                        output_dropped = (
                            0
                            if scene_publisher is None
                            else int(scene_publisher.dropped)
                        )
                        transport_metrics, current_transport_counters = (
                            transport_counter_metrics(
                                loader,
                                scene_publisher,
                                previous_transport_counters,
                            )
                        )
                        logger.info(
                            "Waiting for synchronized Zenoh input: dropped=%s",
                            input_dropped,
                        )
                        if telemetry is not None:
                            telemetry.record_event(
                                {
                                    "event": "waiting_for_input",
                                    "status": "waiting",
                                    "wait_timeout_ms": (
                                        batch_received_at - receive_started_at
                                    )
                                    * 1000.0,
                                    "drops": {
                                        "input_total_by_edge": input_dropped,
                                        "input_delta_by_edge": counter_deltas(
                                            input_dropped,
                                            previous_input_dropped,
                                        ),
                                        "scene_output_total": output_dropped,
                                        "scene_output_delta": max(
                                            0,
                                            output_dropped
                                            - previous_output_dropped,
                                        ),
                                        "telemetry_total": (
                                            telemetry.dropped_records
                                        ),
                                    },
                                    "queues": queue_metrics(
                                        loader,
                                        scene_publisher,
                                        telemetry,
                                    ),
                                    "transport": transport_metrics,
                                    "resource": sample_resources(
                                        resource_sampler
                                    ),
                                }
                            )
                        previous_input_dropped = input_dropped
                        previous_output_dropped = output_dropped
                        previous_transport_counters = (
                            current_transport_counters
                        )
                        last_wait_log = now
                    continue

                batch_count += 1
                persist_batch_metrics = telemetry is not None and (
                    batch_count == 1
                    or batch_count % args.metrics_sample_every == 0
                )
                if batch_count == 1:
                    logger.info("First synchronized batch received")
                started_at = time.perf_counter()
                timings_ms: dict[str, float] = {
                    "receive_wait_and_prepare": (
                        batch_received_at - receive_started_at
                    )
                    * 1000.0,
                }
                if last_batch_received_at is not None:
                    timings_ms["batch_arrival_interval"] = (
                        batch_received_at - last_batch_received_at
                    ) * 1000.0
                last_batch_received_at = batch_received_at
                batch_event: dict[str, Any] = {
                    "batch_index": batch_count,
                    "status": "error",
                    "timings_ms": timings_ms,
                }
                batch_failed = False
                try:
                    heatmaps, roots, reid_items, timestamps, sync_spread = prepared

                    batch_event["source"] = {
                        "edge_timestamps": {
                            edge_id: float(timestamp)
                            for edge_id, timestamp in zip(edge_ids, timestamps)
                        },
                        "sync_spread_ms": float(sync_spread) * 1000.0,
                        "timestamp_kind": (
                            "unix"
                            if timestamps and min(timestamps) >= 946684800
                            else "relative_or_unknown"
                        ),
                    }
                    if persist_batch_metrics:
                        input_by_edge, input_by_camera = count_reid_items(
                            reid_items,
                            edge_ids,
                        )
                        batch_event["data_volume_bytes"] = {
                            "heatmaps": sum(
                                tensor_nbytes(heatmap) for heatmap in heatmaps
                            ),
                            "roots": tensor_nbytes(roots),
                            "reid_features": estimate_reid_feature_bytes(
                                reid_items
                            ),
                        }
                        batch_event["tensor_shapes"] = {
                            "heatmaps": [
                                list(heatmap.shape) for heatmap in heatmaps
                            ],
                            "roots": list(roots.shape),
                        }

                    stage_started_at = time.perf_counter()
                    reid_results = reid.process_realtime(reid_items) or []
                    timings_ms["reid_clustering"] = (
                        time.perf_counter() - stage_started_at
                    ) * 1000.0

                    stage_started_at = time.perf_counter()
                    ids = assign_global_ids(
                        reid_results,
                        roots,
                        edge_ids,
                        edge_metadata,
                    )
                    timings_ms["root_association"] = (
                        time.perf_counter() - stage_started_at
                    ) * 1000.0

                    mapped_ids = {
                        int(global_id)
                        for edge_root_ids in ids
                        for global_id in edge_root_ids
                        if global_id is not None
                    }
                    if persist_batch_metrics:
                        batch_event["reid"] = build_reid_metrics(
                            reid_items,
                            reid_results,
                            mapped_ids,
                            previous_mapped_ids,
                            seen_global_ids,
                            reid,
                            include_observations=(
                                args.metrics_include_reid_observations
                            ),
                        )
                    previous_mapped_ids = mapped_ids
                    seen_global_ids.update(
                        int(item["global_id"]) for item in reid_results
                    )

                    stage_started_at = time.perf_counter()
                    if args.all_lod2:
                        lod_by_id = lod_assign(ids, assign_all_lod2)
                    else:
                        lod_by_id = assign_priority_lods(
                            ids,
                            roots,
                            timestamps,
                            priority_engine,
                            args.lod2_count,
                        )
                    timings_ms["lod_assignment"] = (
                        time.perf_counter() - stage_started_at
                    ) * 1000.0
                    logger.debug("LOD assignments: %s", lod_by_id)

                    stage_started_at = time.perf_counter()
                    if persist_batch_metrics:
                        assert pose_cuda_start is not None
                        assert pose_cuda_end is not None
                        pose_cuda_start.record()
                    pose_result = run_pose_models(
                        pose_models,
                        heatmaps,
                        roots,
                        edge_ids,
                        ids,
                        lod_by_id,
                    )
                    if persist_batch_metrics:
                        pose_cuda_end.record()
                    timings_ms["pose_inference"] = (
                        time.perf_counter() - stage_started_at
                    ) * 1000.0

                    stage_started_at = time.perf_counter()
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
                    timings_ms["scene_build"] = (
                        time.perf_counter() - stage_started_at
                    ) * 1000.0
                    if persist_batch_metrics:
                        # Scene construction has already copied the predictions
                        # to CPU, so the event is complete without an extra
                        # telemetry-only CUDA synchronization.
                        timings_ms["pose_gpu"] = float(
                            pose_cuda_start.elapsed_time(pose_cuda_end)
                        )
                    logger.debug(
                        "Scene output is ready: people=%d",
                        len(scene_output.people),
                    )
                    stage_started_at = time.perf_counter()
                    if viser_debug_output_path is not None:
                        save_scene_for_viser(
                            scene_output,
                            viser_debug_output_path,
                        )
                    timings_ms["viser_debug_output"] = (
                        time.perf_counter() - stage_started_at
                    ) * 1000.0

                    stage_started_at = time.perf_counter()
                    scene_payload = scene_output.to_dict()
                    encoded_scene = encode_scene(scene_payload)
                    timings_ms["scene_serialization"] = (
                        time.perf_counter() - stage_started_at
                    ) * 1000.0

                    stage_started_at = time.perf_counter()
                    zmq_enqueued = None
                    if zmq_output is not None and not zmq_output.send_scene(
                        scene_payload
                    ):
                        zmq_enqueued = False
                        logger.error("Failed to send scene output over ZMQ")
                    elif zmq_output is not None:
                        zmq_enqueued = True
                    zenoh_enqueued = None
                    if (
                        scene_publisher is not None
                        and not scene_publisher.send_payload(encoded_scene)
                    ):
                        zenoh_enqueued = False
                        logger.error("Failed to send scene output over Zenoh")
                    elif scene_publisher is not None:
                        zenoh_enqueued = True
                    timings_ms["transport_enqueue"] = (
                        time.perf_counter() - stage_started_at
                    ) * 1000.0

                    batch_event["workload"] = {
                        "output_people": len(scene_output.people),
                    }
                    if persist_batch_metrics:
                        roots_cpu = roots.detach().cpu()
                        lod_counts = Counter(
                            int(lod) for lod in lod_by_id.values()
                        )
                        batch_event["workload"].update(
                            {
                                "edges": len(edge_ids),
                                "camera_views_per_edge": len(heatmaps),
                                "reid_input_by_edge": input_by_edge,
                                "reid_input_by_edge_camera": input_by_camera,
                                "valid_root_candidates_by_edge": {
                                    edge_id: int(
                                        (roots_cpu[index, :, 3] >= 0).sum()
                                    )
                                    for index, edge_id in enumerate(edge_ids)
                                },
                                "mapped_people_by_edge": {
                                    edge_id: sum(
                                        global_id is not None
                                        for global_id in edge_root_ids
                                    )
                                    for edge_id, edge_root_ids in zip(
                                        edge_ids,
                                        ids,
                                    )
                                },
                                "mapped_people": len(mapped_ids),
                                "lod_counts": {
                                    str(lod): int(lod_counts.get(lod, 0))
                                    for lod in (0, 1, 2)
                                },
                                "pose_targets": int(lod_counts.get(2, 0)),
                                "output_poses": sum(
                                    person.pose is not None
                                    for person in scene_output.people
                                ),
                            }
                        )
                        batch_event["data_volume_bytes"]["scene_json"] = len(
                            encoded_scene
                        )
                    batch_event["transport"] = {
                        "zmq_enqueued": zmq_enqueued,
                        "zenoh_enqueued": zenoh_enqueued,
                    }
                    if persist_batch_metrics:
                        output_wall_time = time.time()
                        batch_event["source"]["source_to_output_age_ms"] = (
                            unix_input_age_ms(timestamps, output_wall_time)
                        )
                    batch_event["status"] = "ok"
                except Exception as exc:
                    batch_failed = True
                    batch_event["error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc)[:2000],
                    }
                    logger.exception("Failed to process synchronized batch")
                finally:
                    elapsed = (time.perf_counter() - started_at) * 1000.0
                    timings_ms["processing_total"] = elapsed
                    input_dropped = {
                        edge_id: int(value)
                        for edge_id, value in loader.subscriber.dropped.items()
                    }
                    output_dropped = (
                        0
                        if scene_publisher is None
                        else int(scene_publisher.dropped)
                    )
                    should_persist_metrics = persist_batch_metrics or batch_failed
                    if should_persist_metrics:
                        pipeline_stage_names = (
                            "reid_clustering",
                            "root_association",
                            "lod_assignment",
                            "pose_inference",
                            "scene_build",
                            "viser_debug_output",
                            "scene_serialization",
                            "transport_enqueue",
                        )
                        pipeline_stages_total = sum(
                            timings_ms.get(name, 0.0)
                            for name in pipeline_stage_names
                        )
                        timings_ms["pipeline_stages_total"] = (
                            pipeline_stages_total
                        )
                        timings_ms["instrumentation_and_unattributed"] = max(
                            0.0,
                            elapsed - pipeline_stages_total,
                        )
                        batch_event["rates"] = {
                            "processing_capacity_fps": (
                                1000.0 / elapsed if elapsed > 0 else None
                            )
                        }
                        batch_event["drops"] = {
                            "input_total_by_edge": input_dropped,
                            "input_delta_by_edge": counter_deltas(
                                input_dropped,
                                previous_input_dropped,
                            ),
                            "scene_output_total": output_dropped,
                            "scene_output_delta": max(
                                0,
                                output_dropped - previous_output_dropped,
                            ),
                            "telemetry_total": (
                                0
                                if telemetry is None
                                else telemetry.dropped_records
                            ),
                        }
                        batch_event["queues"] = queue_metrics(
                            loader,
                            scene_publisher,
                            telemetry,
                        )
                        transport_metrics, current_transport_counters = (
                            transport_counter_metrics(
                                loader,
                                scene_publisher,
                                previous_transport_counters,
                            )
                        )
                        batch_event.setdefault("transport", {}).update(
                            transport_metrics
                        )
                        batch_event["resource"] = sample_resources(
                            resource_sampler
                        )
                        previous_input_dropped = input_dropped
                        previous_output_dropped = output_dropped
                        previous_transport_counters = current_transport_counters
                    if telemetry is not None:
                        telemetry.record_batch(
                            batch_event,
                            persist=should_persist_metrics,
                        )

                if batch_failed:
                    continue
                if (
                    batch_count == 1
                    or batch_count % args.status_log_every == 0
                ):
                    logger.info(
                        "Batch %d complete: inference=%.1fms "
                        "sync_spread=%.1fms people=%d dropped=%s",
                        batch_count,
                        elapsed,
                        sync_spread * 1000,
                        len(scene_output.people),
                        input_dropped,
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
        if telemetry is not None:
            final_transport_metrics, _ = transport_counter_metrics(
                loader,
                scene_publisher,
                previous_transport_counters,
            )
            telemetry.record_event(
                {
                    "event": "shutdown",
                    "status": "ok",
                    "batches_processed": batch_count,
                    "reid_tracker": reid.metrics_snapshot(),
                    "transport": final_transport_metrics,
                    "resource": sample_resources(
                        resource_sampler,
                        force=True,
                    ),
                }
            )
            telemetry.close(
                {
                    "final_input_drops_by_edge": {
                        edge_id: int(value)
                        for edge_id, value in loader.subscriber.dropped.items()
                    },
                    "final_scene_output_drops": (
                        0
                        if scene_publisher is None
                        else int(scene_publisher.dropped)
                    ),
                    "unique_global_ids_observed": len(seen_global_ids),
                }
            )
            logger.info("Inference telemetry saved: %s", telemetry.run_dir)


if __name__ == "__main__":
    main()
