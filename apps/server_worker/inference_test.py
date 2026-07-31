from typing import List, Tuple, Union
import argparse
import asyncio
import time
import json, glob, os, sys, cv2
import torch
import torch.multiprocessing as mp
import numpy as np
import pickle, queue
from datetime import datetime

CWD = os.getcwd()
sys.path.insert(0, CWD)

from apps.server_worker.src.protocol.kafka import Kafka
from torch.utils.data._utils.collate import default_collate
from apps.server_worker.src.utils.edgemetadata import CalibrationData
from apps.server_worker.src.utils.pre_process import PREPROCESS
from apps.server_worker.src.pose.models.multi_person_posenet_ssv import get_multi_person_pose_net
from config.config import config as focus_config
from config.config import update_config as update_focus_config
from apps.server_worker.src.pose.core.config import config as sp3d_config
from apps.server_worker.src.pose.core.config import update_config as update_sp3d_config
# from utils.tensorrt import load_tensorrt_model
from apps.server_worker.src.reid.sliding_clustering import ClusteringSliding
from apps.server_worker.src.protocol.zmq import Protocol

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch AISL Inference")
    parser.add_argument("--cfg_focus", default=None, help="experiment configure file name", type=str)
    parser.add_argument("--tensorrt", action="store_true", default=True, help="If set, the program will use tensorrt.")
    return parser


def convert_root_to_tensor(root_data, zone_idx):
    """Root numpy 데이터를 PyTorch tensor로 변환 (LOD 전처리 포함)"""
    if root_data is None:
        return None

    if isinstance(root_data, np.ndarray):
        # LOD 처리를 numpy 단계에서 수행
        if zone_idx < len(focus_config.LOD) and focus_config.LOD[zone_idx] != 2:
            root_data = root_data.copy()
            root_data[:, 3] = -1
        return torch.from_numpy(root_data).float()
    else:
        return root_data


def convert_heatmap_to_tensor(heatmap_data):
    """Heatmap numpy 데이터를 PyTorch tensor로 변환"""
    if heatmap_data is None:
        return None

    if isinstance(heatmap_data, list):
        tensor_list = []
        for hm in heatmap_data:
            if hm is not None:
                if isinstance(hm, np.ndarray):
                    tensor = torch.from_numpy(hm).to(torch.float32) / 255.0
                    if tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    tensor_list.append(tensor)
                else:
                    tensor_list.append(hm)
        return tensor_list
    else:
        if isinstance(heatmap_data, np.ndarray):
            return torch.from_numpy(heatmap_data).to(torch.float32) / 255.0
        else:
            return heatmap_data


def pickle_data_loader(pickle_file_path):
    """Pickle 파일에서 데이터를 로드하는 함수"""
    print(f"Loading data from pickle file: {pickle_file_path}")
    try:
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} data frames from pickle file")
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


def main():
    # 멀티프로세싱 설정
    mp.set_start_method('spawn', force=True)

    parser = get_parser()
    args = parser.parse_args()

    # ZONE 설정
    zones = focus_config.ZONE if isinstance(focus_config.ZONE, list) else [focus_config.ZONE]

    # 멀티프로세싱 큐 및 이벤트 생성
    stop_event = mp.Event()

    #--------------------------------
    # ReID
    reid = ClusteringSliding(base_dir=focus_config.ReidDataPATH, zones=zones, window_size=10)
    #--------------------------------
    # Pose
    if args.cfg_focus:
        update_focus_config(args.cfg_focus)
    update_sp3d_config(focus_config.POSENET.CONFIG)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.enabled = sp3d_config.CUDNN.ENABLED
        torch.backends.cudnn.benchmark = sp3d_config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = sp3d_config.CUDNN.DETERMINISTIC
        device = torch.device('cuda')
    else:
        raise ValueError('CUDA is not available. Please check your configuration.')

    calibration = CalibrationData(sp3d_config, focus_config, focus_config.CALIBRATION_PATH)
    pose_preprocess = PREPROCESS(sp3d_config)
    pose_model = get_multi_person_pose_net(sp3d_config, inference_mode='posenet')
    pose_model.load_state_dict(torch.load(focus_config.POSENET.CKPT, weights_only=False))
    pose_model = pose_model.eval().to(device)
    # if args.tensorrt:
    #     pose_model = load_tensorrt_model(pose_model, "models/POC_posenet.pth.tar", sp3d_config)

    zmq=Protocol(host=focus_config.ZMQ_SERVER, port=focus_config.ZMQ_PORT)
    # Kafka per zone (송신용 인스턴스도 필요시 사용)
    # zone_kafka = []
    # for i, zone in enumerate(zones):
    #     kafka_instance = Kafka(zone=zone, bootstrap_servers=focus_config.SERVER, auto_offset_reset="latest")
    #     zone_kafka.append(kafka_instance)
    #     kafka_instance.start()
    # print("success load model")

    # Pickle 파일에서 데이터 로드
    pickle_data = pickle_data_loader('/home/min4090/Server_DT/result1.pkl')
    if pickle_data is None:
        print("Failed to load pickle data. Exiting...")
        return
    
    # 데이터 인덱스 초기화
    data_index = 0
    print(f"[MAIN] Loaded pickle data with {len(pickle_data)} frames")

    try:
        while data_index < len(pickle_data):
            # Pickle 데이터에서 현재 프레임 가져오기
            current_frame_data = pickle_data[data_index]
            data_index += 1
            
            if not current_frame_data:
                time.sleep(0.001)
                continue

            # 모든 존 데이터가 존재하는 시점에서만 연산
            root = [None] * len(zones)
            all_heatmaps_data = [None] * len(zones)
            all_heatmaps_zones = [None] * len(zones)
            reid_data = [None] * len(zones)
            current_timestamp = [None] * len(zones)

            # Pickle 데이터는 리스트 형태이므로 각 데이터를 처리
            for data_item in current_frame_data:
                # 첫 번째 zone으로 가정 (pickle 데이터 구조에 따라 조정 필요)
                zone_idx = 0
                zone_name = zones[zone_idx] if zone_idx < len(zones) else zones[0]

                # Root
                root_data = data_item.get("roots")
                root[zone_idx] = convert_root_to_tensor(root_data, zone_idx)

                # Heatmap - pickle 데이터에서는 'allheatmaps' 키 사용
                heatmap_data = data_item.get("allheatmaps")
                if heatmap_data is not None:
                    converted_heatmap = convert_heatmap_to_tensor(heatmap_data)
                    all_heatmaps_data[zone_idx] = converted_heatmap
                    all_heatmaps_zones[zone_idx] = zone_name

                # ReID
                reid_event = data_item.get("reid")
                if reid_event and isinstance(reid_event, list) and len(reid_event) > 0:
                    for item in reid_event:
                        if 'zone' not in item:
                            item['zone'] = zone_name
                        reid_data[zone_idx] = item
                    current_timestamp[zone_idx] = data_item.get('time')
                else:
                    current_timestamp[zone_idx] = None

            # 배치 구성
            valid_heatmaps = [hm for hm in all_heatmaps_data if hm is not None]
            if valid_heatmaps:
                all_heatmap_batch = default_collate(valid_heatmaps)
                if isinstance(all_heatmap_batch, list):
                    all_heatmap_batch = [hm.to(device, non_blocking=True) for hm in all_heatmap_batch]
                else:
                    all_heatmap_batch = all_heatmap_batch.to(device, non_blocking=True)
            else:
                all_heatmap_batch = None

            valid_roots = [r for r in root if r is not None and isinstance(r, torch.Tensor)]
            if valid_roots:
                root_batch = torch.cat(valid_roots, dim=0).to(device, non_blocking=True)
            else:
                root_batch = None

            # ReID
            valid_reid_data = []
            if reid_data:
                for item in reid_data:
                    if item is not None and item.get("feature") is not None:
                        valid_reid_data.append(item)
            # print(f"valid_reid_data: {valid_reid_data}")
            if valid_reid_data:
                result = reid.process_realtime(valid_reid_data)
            else:
                result = None

            # Pose-ID 매핑 및 추론
            if result and all_heatmap_batch is not None and root_batch is not None:
                from apps.server_worker.src.pose.utils import cameras
                ids = [[None] * 10 for _ in range(len(zones))]
                id_root_map_list = []
                roots_2d = []

                # 각 존 root를 2D로 투영
                cams = calibration.get()
                # root_batch는 모든 존 concat 결과이므로, 존 단위 인덱싱이 필요하면 별도 구조를 유지해야 함
                # 여기서는 원 코드 로직 유지
                for i_zone, zone_roots in enumerate(root_batch):
                    if zone_roots is None:
                        continue
                    if i_zone >= len(cams):
                        continue
                    for n in range(len(cams)):
                        zone_roots_tensor = zone_roots[:, :3]
                        if zone_roots_tensor.dim() == 3:
                            zone_roots_tensor = zone_roots_tensor.view(-1, 3)
                        roots_2d.append(cameras.project_pose(zone_roots_tensor, cams[n]))

                global_ids = [None] * len(zones)
                all_global_ids = set([data["global_id"] for data in result])

                for global_id in all_global_ids:
                    id_data = [data for data in result if data["global_id"] == global_id]
                    zones_count = {}
                    for data_item in id_data:
                        zname = data_item["zone"]
                        zones_count[zname] = zones_count.get(zname, 0) + 1
                    if not zones_count:
                        continue
                    selected_zone = max(zones_count, key=zones_count.get)
                    zone_index = zones.index(selected_zone) if selected_zone in zones else 0
                    if zone_index >= len(roots_2d):
                        continue
                    zone_roots_2d = roots_2d[zone_index]
                    cams = calibration.get()
                    if zone_roots_2d is None:
                        continue

                    if zone_roots_2d is not None and len(zone_roots_2d) > 0:
                        zone_len = zone_roots_2d.shape[0] if isinstance(zone_roots_2d, torch.Tensor) else len(zone_roots_2d)
                        all_distances = np.zeros(zone_len)
                        for data_item in id_data:
                            cam_id = data_item["cam"]
                            if cam_id >= len(cams):
                                continue
                            cam = cams[cam_id]
                            bbox = data_item["bbox"]
                            central = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                            zone_roots_2d_cpu = zone_roots_2d.cpu().numpy() if isinstance(zone_roots_2d, torch.Tensor) else zone_roots_2d
                            distances = np.linalg.norm(zone_roots_2d_cpu - np.array(central), axis=1)
                            all_distances += distances
                    else:
                        continue

                    root_idx = np.argmin(all_distances)
                    # selected_zone는 문자열. ids 인덱스 접근을 위해 zone_index 사용
                    ids[zone_index][root_idx] = global_id

                # 추론
                with torch.inference_mode():
                    pose_result = pose_model(
                        input_heatmaps=all_heatmap_batch,
                        grid_centers=root_batch,
                    )

                pose_result["grid_centers"] = pose_result["grid_centers"][:, :, :3].cpu().numpy()
                pose_result["pred"] = pose_result["pred"][:, :, :, :3].cpu().numpy()
                pose_result["id"] = ids
                pose_result["timestamp"] = current_timestamp
                pose_result["zone"] = all_heatmaps_zones

                zmq.send_result(pose_result, focus_config.LOD)
                del all_heatmap_batch, root_batch
                torch.cuda.empty_cache()
            
            
            # 프레임 간 딜레이 추가 (실시간 처리 시뮬레이션)
            time.sleep(0.1)

        print(f"[MAIN] Finished processing all {len(pickle_data)} frames from pickle file")

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        print("[MAIN] Inference completed")


if __name__ == "__main__":
    main()