from multiprocessing import Event,Pipe,shared_memory,Value
import time # 추론 시간 확인용
import sys, cv2, os
import argparse
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from apps.edge_client.src.utils.input import ExampleDataset, IPCamera
from apps.edge_client.src.utils.calibration import CalibrationData
from apps.edge_client.src.utils.pre_process import PREPROCESS
from apps.edge_client.src.utils.transforms import get_transform, get_cam, get_scale, get_affine_transform
from apps.edge_client.src.root.models.multi_person_posenet_ssv import get_multi_person_pose_net
from apps.edge_client.src.protocol.kafka import KafkaProducer, KafkaConsumer
from config.config import config as focus_config
from config.config import update_config as update_focus_config
from apps.edge_client.src.root.core.config import config as sp3d_config
from apps.edge_client.src.root.core.config import update_config as update_sp3d_config
from apps.edge_client.src.utils.tensorrt import export_tensorrt, load_tensorrt_model
from apps.edge_client.src.reid.yolopose import extract_poses_from_frame
from apps.edge_client.src.reid.feature_extract import build_trt_feature_extractor, extract_features_from_persons_trt
import torch.multiprocessing as mp
import traceback
from ultralytics import YOLO

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch AISL Inference")
    parser.add_argument("--cfg_focus", default=None, help="experiment configure file name", type=str)
    parser.add_argument("--example_folder", help="source folder name", type=str)
    parser.add_argument("--path2save", type=str, help="path to save prediction results pickle")
    parser.add_argument("--tensorrt", action="store_true", default=False, help="If set, the program will use tensorrt.")
    parser.add_argument("--pose_mode", type=str, default="posenet", help="posenet, rootnet, heatmap")
    parser.add_argument("--dataset", action="store_true", default=False, help="If set, the program will inference in realtime")
    args, rest = parser.parse_known_args()
    return parser

def cuda_setup():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = sp3d_config.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = sp3d_config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = sp3d_config.CUDNN.DETERMINISTIC
    device = torch.device('cuda')
    arr = torch.tensor([1], device=device) # gpu 메모리 할당 테스트 (빼지 마세요)
    return device

def load_pose_model(cfg:dict, cpkt_path:str, device, tensorrt=False):
    model = get_multi_person_pose_net(cfg, inference_mode="rootnet")
    model.load_state_dict(torch.load(cpkt_path, weights_only=False))
    model = model.eval().to(device)
    if tensorrt:
        model = load_tensorrt_model(model, cpkt_path, cfg, mode="fp16")
    return model

def main():
    parser = get_parser()
    args = parser.parse_args()
    #--------------------------------------------------------------------------------------------------------
    # CUDA 설정
    device = cuda_setup()
    stop_event = Event()
    parent, child = Pipe(duplex=False)
    input_flag = Value('b', True) 
    #Pose 
    if args.cfg_focus:
        update_focus_config(args.cfg_focus)
    update_sp3d_config(focus_config.POSENET.CONFIG)
        # 입력 데이터 소스 설정


    if args.dataset:
        print("Starting synchronized dataset input")
        input_process = ExampleDataset(args.example_folder,stop_event,child,input_flag)
        calibration = CalibrationData(sp3d_config, example_path=args.example_folder)
    else:
        print("Starting synchronized IP camera input")
        input_process=IPCamera(focus_config.CAMERAS,stop_event,child,input_flag)
        calibration = CalibrationData(sp3d_config, cameras=focus_config.CAMERAS)

    input_process.start()
    camera_num = len(focus_config.CAMERAS)
    shms = [None]*camera_num
    shapes = [None]*camera_num
    dtypes = [None]*camera_num
    ready = 0
    while ready < camera_num:
        msg = parent.recv()
        if msg[0] == "OK":
            _, i, name, shape, dtypestr = msg
            shms[i] = shared_memory.SharedMemory(name=name)
            shapes[i] = shape
            dtypes[i] = np.dtype(dtypestr)
            ready += 1
        else:
            _, i = msg
            print(f"카메라 {i} 초기화 실패")

    
    # pose 모델 로드
    pose_preprocess = PREPROCESS(sp3d_config)
    pose_model = load_pose_model(sp3d_config, focus_config.POSENET.CKPT, device, args.tensorrt)
    #------------------------------------------------------------------------------------------------------
    # reid 
    model, context, inputs, outputs, bindings, stream, transform = build_trt_feature_extractor()

    yolo_model = YOLO(focus_config.YOLO.MODEL)
    #------------------------------------------------------------------------------------------------------
    kafka_producer = KafkaProducer(focus_config.ZONE, bootstrap_servers=focus_config.SERVER)
    kafka_producer.connect()
    kafka_consumer = KafkaConsumer(
        bootstrap_servers=focus_config.SERVER,
        input_flag=input_flag,
        auto_offset_reset='latest'
    )
    kafka_consumer.start()
    #------------------------------------------------------------------------------------------------------

    
    # 메인 추론 루프
    try:
        num = 0
        while not stop_event.is_set():
            try:
                # print(f"input_flag.value: {input_flag.value}")
                if not input_flag.value:
                    continue

                image_batchs=[None for i in range(camera_num)] 
                for i in range(camera_num):
                    if shms[i] is None:
                        continue
                    h, w, c = shapes[i]
                    image_batchs[i]=np.ndarray((h, w, c), dtype=dtypes[i], buffer=shms[i].buf)
                timestamp=time.time()

                frame_outputs = []
                #-----------------------------------------------------------
                # reid 추론
                for i in range(len(image_batchs)):
                    try:
                        persons = extract_poses_from_frame(yolo_model,image_batchs[i], zone_id=focus_config.ZONE, cam_id=i+1, frame_num=num)
                        if persons:
                            features = extract_features_from_persons_trt(
                                model,
                                context,
                                inputs,
                                outputs,
                                bindings,
                                stream,
                                transform,
                                persons,
                            )
                            frame_outputs.extend(features)
                    except Exception as e:
                        print(f"ReID inference error at frame {num}, cam {i+1}: {repr(e)}")
                        traceback.print_exc() 
                        
                #-----------------------------------------------------------
                # pose 추론
                
                transed_frames = pose_preprocess(image_batchs)
                _, all_heatmaps, roots = pose_model(views=transed_frames)

                all_heatmaps = [(torch.clamp(heatmap, 0, 1) * 255).round().to(torch.uint8).cpu().numpy() for heatmap in all_heatmaps]
                roots = roots.detach().cpu().numpy()
                #----------------------------------------------------------
                # kafka 전송
                kafka_producer.send(focus_config.OUTPUT_TOPIC,frame_outputs,roots,all_heatmaps,timestamp=timestamp)
                print(f'infer :{time.time()-timestamp}')

                    
                num += 1
                
            except KeyboardInterrupt:
                print("Stopping inference...")
                break
            except Exception as e:
                print(f"Main loop error: {repr(e)}")
                
    finally:
        stop_event.set()

        # 큐/리소스 먼저 정리(있다면)
        for shm in shms:
            if shm is not None:
                shm.close()

        if 'input_process' in locals():
            try:
                # 시작된 프로세스만 처리
                if getattr(input_process, "pid", None) is not None:
                    # 1) 무조건 1차 join으로 시체 수거
                    input_process.join(timeout=3.0)

                    # 2) 아직 살아 있으면 종료 후 재-join
                    if input_process.is_alive():
                        input_process.terminate()
                        input_process.join(timeout=3.0)
            except Exception:
                pass


        cv2.destroyAllWindows()
        print("All processes terminated")

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # safer with torch/cv2
    default_argv = [
        '--example_folder', 'data/data_0705',
        '--tensorrt',
        '--dataset',
        '--pose_mode', 'rootnet'
    ]
    if len(sys.argv) == 1:
        sys.argv.extend(default_argv)
    main()