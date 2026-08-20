# ruff: noqa: E402

import argparse
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
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.edge_client.src.protocol.zenoh import ZenohSender
from apps.edge_client.src.protocol.edge import (
    DEFAULT_IDENTITY_PATH,
    DEFAULT_TOPIC_ROOT,
    EdgeTopics,
    load_edge_metadata,
    load_or_create_edge_id,
)
from apps.edge_client.src.reid.feature_extract import (
    build_trt_feature_extractor,
    extract_features_from_persons_trt,
)
from apps.edge_client.src.reid.yolopose import extract_poses_from_frame
from apps.edge_client.src.root.core.config import config as sp3d_config
from apps.edge_client.src.root.core.config import update_config as update_sp3d_config
from apps.edge_client.src.root.models.multi_person_posenet_ssv import (
    get_multi_person_pose_net,
)
from apps.edge_client.src.utils.calibration import CalibrationData
from apps.edge_client.src.utils.input import ExampleDataset, IPCamera
from apps.edge_client.src.utils.pre_process import PREPROCESS
from apps.edge_client.src.utils.tensorrt import load_tensorrt_model
from apps.edge_client.config.config import config as focus_config
from apps.edge_client.config.config import update_config as update_focus_config
from dt_common.spatial.workspace import build_edge_workspace, load_spatial_context


def parse_args():
    parser = argparse.ArgumentParser(description="Edge inference with Zenoh")
    parser.add_argument("--cfg_focus", type=str)
    parser.add_argument(
        "--deployment",
        help="Shared scene, calibration, edge assignment, and workspace manifest",
    )
    parser.add_argument("--example_folder", type=str)
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument("--dataset", action="store_true")
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
    return parser.parse_args()


def load_models(use_tensorrt):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = sp3d_config.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = sp3d_config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = sp3d_config.CUDNN.DETERMINISTIC

    device = torch.device("cuda")
    torch.tensor([1], device=device)  # CUDA 메모리 할당 확인

    # ProjectLayer copies cfg.TRANSFORM while the pose model is constructed.
    # Build preprocessing first so that the affine transform is available.
    pose_preprocess = PREPROCESS(sp3d_config)
    checkpoint = focus_config.POSENET.CKPT
    pose_model = get_multi_person_pose_net(sp3d_config, inference_mode="rootnet")
    pose_model.load_state_dict(torch.load(checkpoint, weights_only=False))
    pose_model = pose_model.eval().to(device)
    if use_tensorrt:
        pose_model = load_tensorrt_model(
            pose_model, checkpoint, sp3d_config, mode="fp16"
        )

    return (
        pose_preprocess,
        pose_model,
        build_trt_feature_extractor(),
        YOLO(focus_config.YOLO.MODEL),
    )


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
            raise TimeoutError("카메라 공유 메모리 초기화 시간 초과")

        message = parent.recv()
        if message[0] != "OK":
            raise RuntimeError(f"카메라 {message[1]} 초기화 실패")

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
):
    if args.dataset:
        print("Starting synchronized dataset input")
        process = ExampleDataset(args.example_folder, stop_event, child, input_flag)
        if spatial_context is None:
            CalibrationData(sp3d_config, example_path=args.example_folder)
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

    print("Starting synchronized IP camera input")
    if spatial_context is None:
        CalibrationData(sp3d_config, cameras=focus_config.CAMERAS)
    else:
        CalibrationData(
            sp3d_config,
            calibration_path=spatial_context.calibration_path,
            camera_ids=spatial_context.edge_camera_ids[edge_id],
            world_origin_m=spatial_context.world_origin_m,
        )
    return IPCamera(focus_config.CAMERAS, stop_event, child, input_flag)


def extract_reid(
    reid_engine,
    yolo_model,
    images,
    frame_num,
    edge_id,
    camera_ids,
):
    features = []
    for index, image in enumerate(images):
        try:
            persons = extract_poses_from_frame(
                yolo_model,
                image,
                edge_id=edge_id,
                cam_id=camera_ids[index],
                frame_num=frame_num,
            )
            if persons:
                features.extend(
                    extract_features_from_persons_trt(*reid_engine, persons)
                )
        except Exception:
            print(f"ReID error: frame={frame_num}, camera={index + 1}")
            traceback.print_exc()
    return features


def main():
    args = parse_args()
    if args.cfg_focus:
        update_focus_config(args.cfg_focus)
    update_sp3d_config(focus_config.POSENET.CONFIG)
    edge_id = load_or_create_edge_id(args.edge_id_file, args.edge_id)
    spatial_context = (
        load_spatial_context(args.deployment) if args.deployment else None
    )
    sp3d_config.SPATIAL_CONTEXT = spatial_context
    edge_metadata = load_edge_metadata(args.edge_id_file)
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
            args.tensorrt
        )

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
        with torch.inference_mode():
            while not stop_event.is_set():
                started_at = time.time()
                frame_timestamp = (
                    frame_num / input_process.fps if args.dataset else started_at
                )
                image_batches = [
                    np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                    for shm, shape, dtype in buffers
                ]
                frame_outputs = extract_reid(
                    reid_engine,
                    yolo_model,
                    image_batches,
                    frame_num,
                    edge_id,
                    camera_ids,
                )

                views = pose_preprocess(image_batches)
                _, all_heatmaps, roots = pose_model(views=views)
                all_heatmaps = [
                    (heatmap.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                    for heatmap in all_heatmaps
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
                if args.dataset:
                    input_flag.value = True
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
