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
from apps.server_worker.src.utils.calibration import CalibrationData
from apps.server_worker.src.utils.pre_process import PREPROCESS
from apps.server_worker.src.pose.models.multi_person_posenet_ssv import get_multi_person_pose_net
from config.config import config as focus_config
from config.config import update_config as update_focus_config
from apps.server_worker.src.pose.core.config import config as sp3d_config
from apps.server_worker.src.pose.core.config import update_config as update_sp3d_config
from apps.server_worker.src.utils.tensorrt import load_tensorrt_model
from apps.server_worker.src.reid.sliding_clustering import ClusteringSliding
from apps.server_worker.src.protocol.zmq import Protocol
#---------------------------------
# from mesh.utils.smpl import SMPL
# from mesh.utils.graph_utils import build_coarse_graphs
# from mesh.models import pose2mesh_net
# from mesh.utils.funcs_utils import load_checkpoint
#------------------------


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch AISL Inference")
    parser.add_argument("--cfg_focus", default=None, help="experiment configure file name", type=str)
    parser.add_argument("--tensorrt", action="store_true", default=True, help="If set, the program will use tensorrt.")
    return parser

def convert_heatmap_to_tensor(heatmap_data):
    """Heatmap numpy 데이터를 PyTorch tensor로 변환 (최적화)"""
    if heatmap_data is None:
        return None

    if isinstance(heatmap_data, list):
        tensor_list = []
        for hm in heatmap_data:
            if hm is not None:
                if isinstance(hm, np.ndarray):
                    # numpy를 float32로 변환 후 tensor로 변환 (더 효율적)
                    tensor = torch.from_numpy(hm.astype(np.float32))
                    tensor = tensor / 255.0  # 정규화
                    if tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    tensor_list.append(tensor)
                else:
                    tensor_list.append(hm)
        return tensor_list
    else:
        if isinstance(heatmap_data, np.ndarray):
            tensor = torch.from_numpy(heatmap_data.astype(np.float32))
            return tensor / 255.0
        else:
            return heatmap_data


def data_collector_worker(zone, server, data_queue, stop_event):
    """각 zone별 데이터 수집 워커 프로세스"""
    print(f"[WORKER {zone}] Starting data collector worker")
    kafka = None
    try:
        kafka = Kafka(zone=zone, bootstrap_servers=server, auto_offset_reset="latest")
        # Kafka 연결 시도
        if not kafka.connect():
            print(f"[WORKER {zone}] Failed to connect to Kafka")
            return
        print(f"[WORKER {zone}] Successfully connected to Kafka")
        
        while not stop_event.is_set():
            if kafka.poll_messages(timeout_ms=1000):
                data = kafka.get_synchronized_data(timeout=0.01)

                if data:
                    try:

                        data_queue.put(data, timeout=0.1)
                    except Exception:
                        pass
            time.sleep(0.001)
    except Exception as e:
        print(f"[WORKER {zone}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if kafka:
                kafka.close()
        except Exception:
            pass
        print(f"[WORKER {zone}] Data collector worker stopped")


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

#--------------------------------
#pose model
    calibration = CalibrationData(sp3d_config, focus_config, focus_config.CALIBRATION_PATH)
    pose_preprocess = PREPROCESS(sp3d_config)
    pose_model = get_multi_person_pose_net(sp3d_config, inference_mode='posenet')
    pose_model.load_state_dict(torch.load(focus_config.POSENET.CKPT, weights_only=False))
    pose_model = pose_model.eval().to(device)
    
    
#---------------------------------
#mesh model
    # mesh_model = SMPL()
    # graph_Adj, graph_L, graph_perm, graph_perm_reverse = \
    #     build_coarse_graphs(mesh_model.face, focus_config.GRAPH_JOINT_NUM, focus_config.GRAPH_SKELETON, focus_config.GRAPH_FLIP_PAIRS, levels=9)
    # mesh_model = pose2mesh_net.get_model(focus_config.GRAPH_JOINT_NUM, graph_L)

    # checkpoint = load_checkpoint(load_dir=focus_config.MESH.CKPT)
    # mesh_model.load_state_dict(checkpoint['model_state_dict'])
    # mesh_model = mesh_model.eval().to(device)
#---------------------------------

    if args.tensorrt:
        pose_model = load_tensorrt_model(pose_model, "models/POC_posenet.pth.tar", sp3d_config)

    #isaacsim 전달용 프로토콜
    zmq = Protocol(host=focus_config.ZMQ_SERVER, port=focus_config.ZMQ_PORT)

    # 각 zone별 수집 워커 시작
    workers = []
    zone_queue = []
    for zone in zones:
        dq = mp.Queue(maxsize=10)
        worker = mp.Process(
            target=data_collector_worker,
            args=(zone, focus_config.SERVER, dq, stop_event)
        )
        zone_queue.append(dq)
        workers.append(worker)
        worker.start()
        print(f"[MAIN] Started worker for zone: {zone}")

    # 모든 존 데이터가 동시에 준비됐을 때만 처리하기 위한 버퍼

    try:
        while True:
            t0=time.time()
            # 각 존 큐에서 최신 데이터만 버퍼에 유지
            zone_buffers = [None] * len(zones)
            root = [None] * len(zones)
            all_heatmaps_data = [None] * len(zones)
            all_heatmaps_zones = [None] * len(zones)
            reid_data = [None] * len(zones)
            current_timestamp = [None] * len(zones)
            
            idx = 0
            while any(buf is None for buf in zone_buffers):
                try:
                    if zone_buffers[idx] is None:
                        zone_buffers[idx] = zone_queue[idx].get_nowait()

                    # 두 존 모두 데이터가 있을 때마다 시간 동기화 체크
                    if zone_buffers[0] is not None and zone_buffers[1] is not None:
                        time_diff = zone_buffers[0].get('time') - zone_buffers[1].get('time')
                        if time_diff > focus_config.time_diff:
                            zone_buffers[1] = None
                            idx = 1
                        elif time_diff < -focus_config.time_diff:
                            zone_buffers[0] = None
                            idx = 0
                        else:
                            break
                    idx += 1
                    if idx >= len(zones):
                        idx = 0
                except Exception:
                    idx += 1
                    if idx >= len(zones):
                        idx = 0
                    continue

            # 모든 존 데이터가 존재하는 시점에서만 연산
            for zone_idx, data in enumerate(zone_buffers):
                zone_name = zones[zone_idx]

                # Root
                root_data = data.get("roots")
                if zone_idx < len(focus_config.LOD) and focus_config.LOD[zone_idx] < 2:
                    # copy 대신 view 사용하여 메모리 효율성 향상
                    root_data = root_data.copy()  # LOD 수정이 필요하므로 copy 필요
                    root_data[:, 3] = -1
                root[zone_idx] = torch.from_numpy(root_data.astype(np.float32))

                # Heatmap
                heatmap_data = data.get("heatmaps")
                if heatmap_data is not None:
                    converted_heatmap = convert_heatmap_to_tensor(heatmap_data)
                    all_heatmaps_data[zone_idx] = converted_heatmap
                    all_heatmaps_zones[zone_idx] = zone_name

                # ReID
                reid_event = data.get("reid")
                if reid_event and isinstance(reid_event, list) and len(reid_event) > 0:
                    for item in reid_event:
                        item['zone'] = zone_name
                        reid_data[zone_idx] = item
                    current_timestamp[zone_idx] = data.get('time')
                else:
                    current_timestamp[zone_idx] = None
            
            # 배치 구성
            valid_heatmaps = [hm for hm in all_heatmaps_data if hm is not None]

            if valid_heatmaps:
                # (N,B,15,128,256) 텐서 만들기
                batch = torch.stack(
                    [torch.stack(sample, dim=0) for sample in valid_heatmaps],  # 각 sample: (B,15,128,256)
                    dim=0
                )  # (N,B,15,128,256)

                # GPU로 한 번에 전송
                batch = batch.to(device, non_blocking=True)

                # (B,N,15,128,256)로 축 변환
                batch = batch.permute(1, 0, 2, 3, 4).contiguous()

                # 리스트로 쪼개기: [tensor(2,15,128,256), ...] 길이 B
                all_heatmap_batch = list(batch.unbind(dim=0))
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
            if valid_reid_data:
                reid_results = reid.process_realtime(valid_reid_data)
            else:
                reid_results = None

            # Pose-ID 매핑 및 추론
            # root_batch가 None이 아닌지 확인 (tensor이므로 is not None로 체크)
            has_valid_roots = root_batch is not None
            
            if reid_results and all_heatmap_batch is not None and has_valid_roots:
                from apps.server_worker.src.pose.utils import cameras
                ids = [[None] * 10 for _ in range(len(zones))]
                id_root_map_list = []
                roots_2d = []

                # 각 존 root를 2D로 투영
                cams = calibration.get()
                # root는 zone별 리스트, root_batch는 모든 존 concat 결과
                # 존 단위 인덱싱을 위해 원래 root 리스트 사용
                for i_zone, zone_roots in enumerate(root):
                    if zone_roots is None:
                        continue
                    if i_zone >= len(cams):
                        continue
                    # zone_roots가 tensor가 아니면 스킵
                    if not isinstance(zone_roots, torch.Tensor):
                        continue
                    for n in range(len(cams)):
                        zone_roots_tensor = zone_roots[:, :3]
                        if zone_roots_tensor.dim() == 3:
                            zone_roots_tensor = zone_roots_tensor.view(-1, 3)
                        roots_2d.append(cameras.project_pose(zone_roots_tensor, cams[n]))

                global_ids = [None] * len(zones)
                all_global_ids = set([data["global_id"] for data in reid_results])

                for global_id in all_global_ids:
                    id_data = [data for data in reid_results if data["global_id"] == global_id]
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
#--------------------------------
#개선 많이 필요
                # if max(focus_config.LOD)==3:
                #     zmq.send_result(pose_result,ids, current_timestamp, all_heatmaps_zones,focus_config.LOD)
                #     for zone_idx, lod in enumerate(focus_config.LOD):
                #         if lod !=3:
                #             pose_result["pred"][zone_idx][:,:,:,3] =-1 # not person
                #     mesh_result,pose_result["pred"]=mesh_model(pose_result["pred"],cams)
                #     pose_result["mesh"]=mesh_result
#----------------------------------
                zmq.send_result(pose_result,ids,current_timestamp, all_heatmaps_zones,focus_config.LOD)
                print(f"zmq time: {time.time()-t0}")
                del all_heatmap_batch, root_batch
                torch.cuda.empty_cache()

            # 한 세트 처리 후 버퍼 초기화. 다음 “동시 세트”를 기다림
            zone_buffers = [None] * len(zones)

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        print("[MAIN] Stopping worker processes...")
        stop_event.set()
        for worker in workers:
            try:
                worker.join(timeout=5)
                if worker.is_alive():
                    print(f"[MAIN] Force terminating worker {worker.pid}")
                    worker.terminate()
            except Exception as e:
                print(f"[MAIN] Error stopping worker: {e}")
        print("[MAIN] All workers stopped")


if __name__ == "__main__":
    main()