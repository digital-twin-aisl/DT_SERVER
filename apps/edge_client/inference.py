# ruff: noqa: E402

import argparse
import hashlib
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import pickle
import struct
import sys
import time
import traceback
import zstandard as zstd

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDGE_CLIENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.edge_client.src.protocol.zenoh import ZenohSender
from apps.edge_client.src.protocol.edge import (
    DEFAULT_IDENTITY_PATH,
    DEFAULT_TOPIC_ROOT,
    EdgeTopics,
    load_edge_metadata,
    load_or_create_edge_id,
)
from apps.edge_client.src.reid.yolopose import extract_poses_from_frames
from apps.edge_client.src.root.core.config import config as sp3d_config
from apps.edge_client.src.root.core.config import update_config as update_sp3d_config
from apps.edge_client.src.root.models.multi_person_posenet_ssv import (
    get_multi_person_pose_net,
)
from apps.edge_client.src.utils.calibration import CalibrationData
from apps.edge_client.src.utils.input import (
    ExampleDataset,
    FrameUnavailableError,
    IPCamera,
    find_dataset_calibration,
    load_camera_sources,
    select_dataset_camera_ids,
    select_live_cameras,
    snapshot_shared_frames,
)
from apps.edge_client.src.utils.pre_process import PREPROCESS
from apps.edge_client.src.utils.tensorrt import load_tensorrt_model
from apps.edge_client.config.config import config as focus_config
from apps.edge_client.config.config import update_config as update_focus_config
from dt_common.spatial.workspace import build_edge_workspace, load_spatial_context


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Edge inference with Zenoh")
    parser.add_argument("--cfg-focus", "--cfg_focus", dest="cfg_focus")
    parser.add_argument(
        "--deployment",
        help="Shared scene, calibration, edge assignment, and workspace manifest",
    )
    parser.add_argument("--example-folder", "--example_folder", dest="example_folder")
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument(
        "--rebuild-tensorrt",
        action="store_true",
        help="Force regeneration of the configuration-specific pose engines",
    )
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument(
        "--camera-config",
        help="Edge-private YAML containing live RTSP camera sources",
    )
    parser.add_argument(
        "--rtsp-transport",
        choices=("tcp", "udp"),
        default="tcp",
    )
    parser.add_argument("--rtsp-open-timeout-ms", type=int, default=8000)
    parser.add_argument("--rtsp-read-timeout-ms", type=int, default=3000)
    parser.add_argument("--rtsp-reconnect-delay", type=float, default=0.5)
    parser.add_argument("--rtsp-max-frame-age", type=float, default=2.0)
    parser.add_argument("--rtsp-max-skew", type=float, default=0.5)
    parser.add_argument("--zenoh-endpoint", type=str)
    parser.add_argument("--zenoh-config", type=str)
    parser.add_argument("--zenoh-topic", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--edge-id", type=str)
    parser.add_argument("--edge-id-file", default=str(DEFAULT_IDENTITY_PATH))
    parser.add_argument("--topic-root")
    parser.add_argument(
        "--no-zenoh",
        action="store_true",
        help="Run inference without publishing results to Zenoh",
    )
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Skip YOLO/ReID extraction and run only multi-view 3D inference",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many frames (useful for dataset smoke tests)",
    )
    args = parser.parse_args(argv)
    if args.dataset and not args.example_folder:
        parser.error("--dataset requires --example-folder")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if min(args.rtsp_open_timeout_ms, args.rtsp_read_timeout_ms) < 1:
        parser.error("RTSP timeouts must be positive")
    if min(
        args.rtsp_reconnect_delay,
        args.rtsp_max_frame_age,
        args.rtsp_max_skew,
    ) <= 0:
        parser.error("RTSP reconnect/freshness values must be positive")
    return args


def resolve_edge_path(value, *, must_exist=True):
    """Resolve app-relative defaults while preserving explicit cwd-relative paths."""
    path = Path(value).expanduser()
    candidates = (
        [path]
        if path.is_absolute()
        else [Path.cwd() / path, EDGE_CLIENT_ROOT / path]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    resolved = candidates[-1].resolve()
    if must_exist:
        raise FileNotFoundError(f"required path does not exist: {resolved}")
    return resolved


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_spatial_context(dataset_path, spatial_context):
    """Prevent absolute-USD datasets from using the legacy fixed workspace."""
    calibration_path = find_dataset_calibration(dataset_path)
    if calibration_path is None:
        return
    if spatial_context is None:
        raise ValueError(
            "a dataset with calibration_result*.json requires --deployment; "
            "the deployment builds its Ground/AOI workspace instead of using "
            "the legacy 80x80x20 cube"
        )
    calibration_sha256 = file_sha256(calibration_path)
    if calibration_sha256 != spatial_context.calibration_sha256:
        raise ValueError(
            "dataset calibration does not match the deployment calibration: "
            f"{calibration_path}"
        )


def resolve_live_cameras(args, edge_metadata, spatial_context, edge_id):
    """Resolve one ordered RTSP source for every edge camera ID."""
    identity_camera_ids = edge_metadata.get("camera_ids")
    deployment_camera_ids = (
        None
        if spatial_context is None
        else list(spatial_context.edge_camera_ids[edge_id])
    )
    if identity_camera_ids is not None:
        identity_camera_ids = [int(value) for value in identity_camera_ids]
        if (
            deployment_camera_ids is not None
            and identity_camera_ids != deployment_camera_ids
        ):
            raise ValueError(
                f"identity camera_ids {identity_camera_ids} do not match "
                f"deployment camera_ids {deployment_camera_ids} for {edge_id}"
            )
    required_camera_ids = deployment_camera_ids or identity_camera_ids

    camera_config_value = args.camera_config or edge_metadata.get("camera_config")
    if camera_config_value:
        camera_config_path = Path(camera_config_value).expanduser()
        if not camera_config_path.is_absolute():
            if args.camera_config:
                camera_config_path = resolve_edge_path(camera_config_path)
            else:
                camera_config_path = (
                    Path(args.edge_id_file).parent / camera_config_path
                ).resolve()
        if not camera_config_path.is_file():
            raise FileNotFoundError(
                f"camera config not found: {camera_config_path}"
            )
        cameras = load_camera_sources(camera_config_path)
    else:
        cameras = focus_config.CAMERAS
    selected = select_live_cameras(
        cameras,
        required_camera_ids,
        sp3d_config.NUM_VIEWS,
    )
    print(
        "Selected live RTSP cameras: "
        + ", ".join(str(camera["id"]) for camera in selected)
    )
    return selected


def load_models(use_tensorrt, use_reid, rebuild_tensorrt=False):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = sp3d_config.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = sp3d_config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = sp3d_config.CUDNN.DETERMINISTIC

    device = torch.device("cuda")
    torch.tensor([1], device=device)  # CUDA 메모리 할당 확인

    # ProjectLayer copies cfg.TRANSFORM while the pose model is constructed.
    # Build preprocessing first so that the affine transform is available.
    pose_preprocess = PREPROCESS(sp3d_config)
    checkpoint = resolve_edge_path(focus_config.POSENET.CKPT)
    pose_model = get_multi_person_pose_net(sp3d_config, inference_mode="rootnet")
    checkpoint_state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    checkpoint_state = {
        name: value
        for name, value in checkpoint_state.items()
        if not name.startswith("pose_net.")
    }
    pose_model.load_state_dict(checkpoint_state)
    del checkpoint_state
    pose_model = pose_model.eval().to(device)
    if use_tensorrt:
        pose_model = load_tensorrt_model(
            pose_model,
            checkpoint,
            sp3d_config,
            mode="fp16",
            force_rebuild=rebuild_tensorrt,
        )

    reid_engine = None
    yolo_model = None
    if use_reid:
        from apps.edge_client.src.reid.feature_extract import (
            build_feature_extractor,
            build_trt_feature_extractor,
        )

        if use_tensorrt:
            reid_engine = (
                "tensorrt",
                build_trt_feature_extractor(
                    force_rebuild=rebuild_tensorrt,
                ),
            )
        else:
            reid_engine = ("pytorch", build_feature_extractor())
        yolo_checkpoint = resolve_edge_path(focus_config.YOLO.MODEL)
        if use_tensorrt:
            from apps.edge_client.src.reid.yolo_tensorrt import build_trt_yolo

            yolo_model = build_trt_yolo(
                yolo_checkpoint,
                batch_size=sp3d_config.NUM_VIEWS,
                image_size=int(focus_config.YOLO.IMAGE_SIZE),
                force_rebuild=rebuild_tensorrt,
            )
        else:
            from ultralytics import YOLO

            yolo_model = YOLO(str(yolo_checkpoint))

    return pose_preprocess, pose_model, reid_engine, yolo_model


PAYLOAD_HEADER = struct.Struct("!4sd")
PAYLOAD_MAGIC = b"ZNH1"
COMPRESSOR = zstd.ZstdCompressor(level=1)


def serialize_output(
    timestamp,
    reid,
    roots,
    heatmaps,
    camera_ids,
    spatial_identity=None,
):
    output = {
        "time": timestamp,
        "reid": reid,
        "roots": roots,
        "allheatmaps": heatmaps,
        "camera_ids": list(camera_ids),
    }
    if spatial_identity is not None:
        output["spatial_context"] = dict(spatial_identity)
    compressed = COMPRESSOR.compress(
        pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)
    )
    return PAYLOAD_HEADER.pack(PAYLOAD_MAGIC, timestamp) + compressed


def attach_input_buffers(parent, process):
    buffers = [None] * len(process.paths)

    for _ in buffers:
        if not parent.poll(15):
            if not process.is_alive():
                raise RuntimeError(
                    "input process exited during camera initialization"
                )
            raise TimeoutError("카메라 공유 메모리 초기화 시간 초과")

        message = parent.recv()
        if message[0] != "OK":
            _, index, *details = message
            camera_id = details[0] if details else index
            reason = details[1] if len(details) > 1 else "unknown error"
            raise RuntimeError(
                f"camera {camera_id} initialization failed: {reason}"
            )

        _, index, name, shape, dtype = message
        buffers[index] = (
            shared_memory.SharedMemory(name=name),
            shape,
            np.dtype(dtype),
        )

    if not process.is_alive():
        raise RuntimeError("입력 프로세스가 초기화 중 종료되었습니다")
    return buffers


def stop_process(process, timeout=3):
    if process is None or process.pid is None:
        return
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=timeout)


def create_input_process(
    args,
    stop_event,
    child,
    input_flag,
    edge_id,
    spatial_context,
    live_cameras=None,
):
    if args.dataset:
        print("Starting synchronized dataset input")
        camera_ids = (
            list(spatial_context.edge_camera_ids[edge_id])
            if spatial_context is not None
            else select_dataset_camera_ids(
                args.example_folder,
                edge_id,
                sp3d_config.NUM_VIEWS,
            )
        )
        process = ExampleDataset(
            args.example_folder,
            stop_event,
            child,
            input_flag,
            camera_ids=camera_ids,
        )
        if spatial_context is None:
            CalibrationData(
                sp3d_config,
                example_path=args.example_folder,
                camera_ids=camera_ids,
            )
        else:
            expected_ids = list(spatial_context.edge_camera_ids[edge_id])
            if process.camera_ids != expected_ids:
                raise ValueError(
                    f"dataset camera order mismatch for {edge_id}: "
                    f"expected {expected_ids}, got {process.camera_ids}"
                )
            CalibrationData(
                sp3d_config,
                calibration_path=spatial_context.calibration_path,
                camera_ids=expected_ids,
                world_origin_m=spatial_context.world_origin_m,
            )
        return process

    print("Starting live RTSP camera input")
    if not live_cameras:
        raise ValueError("live RTSP input requires configured camera sources")
    if spatial_context is None:
        CalibrationData(sp3d_config, cameras=live_cameras)
    else:
        CalibrationData(
            sp3d_config,
            calibration_path=spatial_context.calibration_path,
            camera_ids=spatial_context.edge_camera_ids[edge_id],
            world_origin_m=spatial_context.world_origin_m,
        )
    return IPCamera(
        live_cameras,
        stop_event,
        child,
        input_flag,
        transport=args.rtsp_transport,
        open_timeout_ms=args.rtsp_open_timeout_ms,
        read_timeout_ms=args.rtsp_read_timeout_ms,
        reconnect_delay=args.rtsp_reconnect_delay,
        expected_size=sp3d_config.NETWORK.IMAGE_SIZE_ORIG,
    )


def extract_reid(
    reid_engine,
    yolo_model,
    images,
    frame_num,
    edge_id,
    camera_ids,
):
    if reid_engine is None or yolo_model is None:
        return []
    from apps.edge_client.src.reid.feature_extract import (
        extract_features_from_persons,
        extract_features_from_persons_trt,
    )

    try:
        persons_by_camera = extract_poses_from_frames(
            yolo_model,
            images,
            edge_id=edge_id,
            camera_ids=camera_ids,
            frame_num=frame_num,
        )
        persons = [
            person
            for camera_persons in persons_by_camera
            for person in camera_persons
        ]
        if not persons:
            return []
        backend, extractor = reid_engine
        if backend == "tensorrt":
            return extract_features_from_persons_trt(*extractor, persons)
        return extract_features_from_persons(*extractor, persons)
    except Exception:
        print(f"ReID error: frame={frame_num}")
        traceback.print_exc()
        return []


def main():
    args = parse_args()
    if args.cfg_focus:
        update_focus_config(resolve_edge_path(args.cfg_focus))
    update_sp3d_config(resolve_edge_path(focus_config.POSENET.CONFIG))
    use_tensorrt = args.tensorrt or bool(focus_config.POSENET.TENSORRT)
    if args.dataset:
        args.example_folder = str(resolve_edge_path(args.example_folder))
    if args.deployment:
        args.deployment = str(resolve_edge_path(args.deployment))
    args.edge_id_file = str(
        resolve_edge_path(args.edge_id_file, must_exist=False)
    )
    if args.zenoh_config:
        args.zenoh_config = str(resolve_edge_path(args.zenoh_config))
    edge_id = load_or_create_edge_id(args.edge_id_file, args.edge_id)
    spatial_context = (
        load_spatial_context(args.deployment) if args.deployment else None
    )
    if args.dataset:
        validate_dataset_spatial_context(args.example_folder, spatial_context)
    sp3d_config.SPATIAL_CONTEXT = spatial_context
    edge_metadata = load_edge_metadata(args.edge_id_file)
    live_cameras = (
        None
        if args.dataset
        else resolve_live_cameras(
            args,
            edge_metadata,
            spatial_context,
            edge_id,
        )
    )
    topic_root = args.topic_root or edge_metadata.get("topic_root") or DEFAULT_TOPIC_ROOT
    zenoh_endpoint = (
        args.zenoh_endpoint
        or edge_metadata.get("zenoh_endpoint")
        or focus_config.SERVER
    )
    inference_topic = args.zenoh_topic or EdgeTopics(
        edge_id,
        topic_root,
    ).inference

    stop_event = mp.Event()
    input_flag = mp.Value("b", not args.dataset)
    parent, child = mp.Pipe(duplex=False)
    input_process = None
    sender = None
    buffers = []

    try:
        input_process = create_input_process(
            args,
            stop_event,
            child,
            input_flag,
            edge_id,
            spatial_context,
            live_cameras,
        )
        if spatial_context is not None:
            expected_ids = list(spatial_context.edge_camera_ids[edge_id])
            if input_process.camera_ids != expected_ids:
                raise ValueError(
                    f"input camera order mismatch for {edge_id}: "
                    f"expected {expected_ids}, got {input_process.camera_ids}"
                )
            workspace = build_edge_workspace(
                spatial_context,
                edge_id,
                sp3d_config.CAMS,
                sp3d_config.NETWORK.IMAGE_SIZE_ORIG,
                sp3d_config.MULTI_PERSON.SPACE_SIZE,
                sp3d_config.MULTI_PERSON.INITIAL_CUBE_SIZE,
            )
            sp3d_config.EDGE_WORKSPACE = workspace
            spatial_identity = {
                **spatial_context.identity(),
                "edge_id": edge_id,
                "workspace_id": workspace.workspace_id,
            }
            print(
                f"Workspace ready: source={workspace.source} "
                f"size_m={(workspace.size_mm / 1000.0).round(3).tolist()} "
                f"cube={workspace.cube_size.tolist()} "
                f"valid={int(workspace.valid_mask.sum())}/"
                f"{workspace.valid_mask.size}"
            )
        else:
            sp3d_config.EDGE_WORKSPACE = None
            spatial_identity = None
        input_process.start()
        child.close()

        buffers = attach_input_buffers(parent, input_process)
        camera_ids = input_process.camera_ids

        pose_preprocess, pose_model, reid_engine, yolo_model = load_models(
            use_tensorrt,
            not args.no_reid,
            args.rebuild_tensorrt,
        )
        # TensorRT synchronizes around enqueueV3 when invoked on CUDA's
        # default stream.  Keep the whole inference chain on one dedicated
        # stream so dependencies remain ordered without those global syncs.
        inference_stream = torch.cuda.Stream()
        inference_stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_stream(inference_stream)
        reid_stream = torch.cuda.Stream()
        reid_stream.wait_stream(inference_stream)
        if args.no_reid:
            print("ReID output disabled")

        if args.no_zenoh:
            print("Zenoh output disabled")
        else:
            sender = ZenohSender(
                topic=inference_topic,
                endpoint=zenoh_endpoint,
                config_path=args.zenoh_config,
                queue_size=2,
            )

        frame_num = 0
        last_input_warning = 0.0
        with torch.inference_mode():
            while not stop_event.is_set():
                started_at = time.time()
                frame_timestamp = (
                    frame_num / input_process.fps if args.dataset else started_at
                )
                try:
                    image_batches = snapshot_shared_frames(
                        buffers,
                        input_process.frame_locks,
                        (
                            None
                            if args.dataset
                            else input_process.frame_timestamps
                        ),
                        max_age=(
                            None if args.dataset else args.rtsp_max_frame_age
                        ),
                        max_skew=(
                            None if args.dataset else args.rtsp_max_skew
                        ),
                    )
                except FrameUnavailableError as exc:
                    now = time.monotonic()
                    if now - last_input_warning >= 2.0:
                        print(f"Live input waiting: {exc}")
                        last_input_warning = now
                    time.sleep(0.01)
                    continue
                if args.dataset:
                    # The snapshot above owns copies, so decode the next
                    # synchronized frame while this one is on the GPU.
                    input_flag.value = True

                views = pose_preprocess(image_batches)
                _, all_heatmaps, roots = pose_model(views=views)
                with torch.cuda.stream(reid_stream):
                    frame_outputs = extract_reid(
                        reid_engine,
                        yolo_model,
                        image_batches,
                        frame_num,
                        edge_id,
                        camera_ids,
                    )

                heatmap_batch = (
                    torch.cat(all_heatmaps, dim=0)
                    .clamp(0, 1)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                all_heatmaps = [
                    heatmap_batch[index : index + 1]
                    for index in range(len(camera_ids))
                ]
                roots = roots.detach().cpu().numpy()

                payload = serialize_output(
                    frame_timestamp,
                    frame_outputs,
                    roots,
                    all_heatmaps,
                    camera_ids,
                    spatial_identity,
                )
                if sender is not None:
                    sender.send(payload)

                elapsed = time.time() - started_at
                print(
                    f"frame={frame_num} infer={elapsed:.3f}s "
                    f"payload={len(payload) / 1024**2:.2f}MB "
                    f"dropped={sender.dropped if sender is not None else 0}"
                )
                frame_num += 1
                if args.max_frames is not None and frame_num >= args.max_frames:
                    break
                if args.dataset:
                    while input_flag.value and input_process.is_alive():
                        time.sleep(0.001)
                    if not input_process.is_alive():
                        break

    except KeyboardInterrupt:
        print("Stopping inference...")
    finally:
        stop_event.set()

        if sender is not None:
            sender.close()
        for shm, _, _ in buffers:
            shm.close()

        parent.close()
        child.close()
        stop_process(input_process)
        cv2.destroyAllWindows()
        print("All processes terminated")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
