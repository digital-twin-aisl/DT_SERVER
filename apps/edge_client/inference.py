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


def parse_args():
    parser = argparse.ArgumentParser(description="Edge inference with Zenoh")
    parser.add_argument("--cfg_focus", type=str)
    parser.add_argument("--example_folder", type=str)
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--zenoh-endpoint", type=str)
    parser.add_argument("--zenoh-config", type=str)
    parser.add_argument("--zenoh-topic", type=str)
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

    checkpoint = focus_config.POSENET.CKPT
    pose_model = get_multi_person_pose_net(sp3d_config, inference_mode="rootnet")
    pose_model.load_state_dict(torch.load(checkpoint, weights_only=False))
    pose_model = pose_model.eval().to(device)
    if use_tensorrt:
        pose_model = load_tensorrt_model(
            pose_model, checkpoint, sp3d_config, mode="fp16"
        )

    return (
        PREPROCESS(sp3d_config),
        pose_model,
        build_trt_feature_extractor(),
        YOLO(focus_config.YOLO.MODEL),
    )


PAYLOAD_HEADER = struct.Struct("!4sd")
PAYLOAD_MAGIC = b"ZNH1"
COMPRESSOR = zstd.ZstdCompressor(level=1)


def serialize_output(timestamp, reid, roots, heatmaps):
    output = {
        "time": timestamp,
        "reid": reid,
        "roots": roots,
        "allheatmaps": heatmaps,
    }
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


def create_input_process(args, stop_event, child, input_flag):
    if args.dataset:
        print("Starting synchronized dataset input")
        CalibrationData(sp3d_config, example_path=args.example_folder)
        return ExampleDataset(args.example_folder, stop_event, child, input_flag)

    print("Starting synchronized IP camera input")
    CalibrationData(sp3d_config, cameras=focus_config.CAMERAS)
    return IPCamera(focus_config.CAMERAS, stop_event, child, input_flag)


def extract_reid(reid_engine, yolo_model, images, frame_num):
    features = []
    for index, image in enumerate(images):
        try:
            persons = extract_poses_from_frame(
                yolo_model,
                image,
                edge_id=focus_config.OUTPUT_TOPIC,
                cam_id=index + 1,
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

    stop_event = mp.Event()
    input_flag = mp.Value("b", True)  # 입력 프로세스 API 호환용
    parent, child = mp.Pipe(duplex=False)
    input_process = None
    sender = None
    buffers = []

    try:
        input_process = create_input_process(args, stop_event, child, input_flag)
        input_process.start()
        child.close()

        buffers = attach_input_buffers(parent, input_process)

        pose_preprocess, pose_model, reid_engine, yolo_model = load_models(
            args.tensorrt
        )

        if args.no_zenoh:
            print("Zenoh output disabled")
        else:
            sender = ZenohSender(
                topic=args.zenoh_topic or focus_config.OUTPUT_TOPIC,
                endpoint=args.zenoh_endpoint or focus_config.SERVER,
                config_path=args.zenoh_config,
                queue_size=2,
            )

        frame_num = 0
        with torch.inference_mode():
            while not stop_event.is_set():
                started_at = time.time()
                image_batches = [
                    np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                    for shm, shape, dtype in buffers
                ]
                frame_outputs = extract_reid(
                    reid_engine, yolo_model, image_batches, frame_num
                )

                views = pose_preprocess(image_batches)
                _, all_heatmaps, roots = pose_model(views=views)
                all_heatmaps = [
                    (heatmap.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                    for heatmap in all_heatmaps
                ]
                roots = roots.detach().cpu().numpy()

                payload = serialize_output(
                    started_at, frame_outputs, roots, all_heatmaps
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
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--example_folder",
                "data/data_0705",
                "--tensorrt",
                "--dataset",
            ]
        )
    main()
