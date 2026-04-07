import asyncio
import sys
import os
import grpc
import time
import json

# 현재 디렉토리(edge_client)에 있는 생성된 protobuf 파일 참조
sys.path.append(os.path.dirname(__file__))

try:
    import apps.edge_client.edge_communication_pb2 as edge_communication_pb2
    import apps.edge_client.edge_communication_pb2_grpc as edge_communication_pb2_grpc
except ImportError as e:
    print(f"[오류] Protobuf 모듈을 불러오지 못했습니다. 먼저 Protoc 컴파일러를 실행해주세요: {e}")
    sys.exit(1)

TARGET_EDGE_ID = "edge_001"

# --- 엣지 구동을 위한 모의(Mock) 클래스 선언 ---
# 실제 운영 시, models_temp 하위의 inference 모듈이나 현장 카메라 제어 코드로 대체합니다.
class MultiCameraManager:
    def __init__(self, configs):
        self.configs = configs
    def start_all(self):
        pass
    def get_synchronized_frames(self, tolerance):
        return {"timestamp": time.time(), "frames": [None, None, None]}
    def stop_all(self):
        pass

class TensorRTInferenceEngine:
    def __init__(self, model_path):
        pass
    def infer(self, frames):
        return b"dummy_tensor_bytes_generated_by_jetson"

class ONVIFAdapter:
    def register_camera(self, *args):
        pass
    def handle_command(self, cam_id, command, params):
        print(f"   [ONVIF] 엣지 장치 로컬명령: {cam_id} 에 '{command}' 실행 (params: {params})")

async def generate_requests():
    """
    서버로 데이터를 올려보내는(Uplink) 제너레이터 함수.
    양방향 스트리밍의 무한 루프를 유지하면서 일정 주기마다 데이터를 보냄.
    """
    print(f"[{TARGET_EDGE_ID}] Uplink 스트림이 시작되었습니다. 첫 메시지로 장치를 등록합니다.")
    
    # 1. 서버에 내 device_id가 이거라고 처음 알려주는 역할
    yield edge_communication_pb2.EdgeStreamRequest(
        device_id=TARGET_EDGE_ID,
        heartbeat="INIT_CONNECTION"
    )
    
    # 2. 이후 5초마다 가짜 VideoData를 전송하여 연결 유지 및 데이터 송신
    # 3대 카메라 임시 설정 (시나리오) - RTSP URL은 현장 상황에 맞게 변경
    camera_configs = {
        "cam_01": "rtsp://admin:byda13245@10.1.2.10:554/1",
        "cam_02": "rtsp://localhost:8554/cam2",
        "cam_03": "rtsp://localhost:8554/cam3"
    }

    # 현장 엣지용 딥러닝 추론 엔진 및 카메라 매니저 초기화 (위에 정의한 모의 클래스 사용)
    # 실제로는 import models_temp.inference 등으로 현장용 모듈을 사용합니다.
    cam_manager = MultiCameraManager(camera_configs)
    cam_manager.start_all()
    inference_engine = TensorRTInferenceEngine(model_path="yolov8n.engine")
    
    frame_idx = 0
    try:
        while True:
            # 약간의 대기시간 (약 10 FPS 전송 제한)
            await asyncio.sleep(0.1)
            
            # 동기화된 멀티 카메라 프레임 세트 획득
            sync_data = cam_manager.get_synchronized_frames(tolerance=0.05)
            if not sync_data:
                continue # 동기화된 프레임을 아직 찾지 못했거나 버퍼 부족시 패스
            
            frame_idx += 1
            # 초 단위에서 밀리초(ms) timestamp로 변환
            timestamp = int(sync_data["timestamp"] * 1000)
            frames = sync_data["frames"]
            
            # [전처리 & 로컬 추론] Jetson에서 TensorRT로 특징 벡터 직렬화
            tensor_bytes = inference_engine.infer(frames)
            
            if frame_idx % 30 == 0:
                print(f"[EDGE -> SERVER] 실시간 특징 텐서 전송 중... (frame_id: {frame_idx}, size: {len(tensor_bytes)} bytes)")
            
            # Protobuf 규격에 맞게 포장
            video_data = edge_communication_pb2.VideoFeatureData(
                timestamp=timestamp,
                frame_id=str(frame_idx),
                tensor_data=tensor_bytes
            )
            
            yield edge_communication_pb2.EdgeStreamRequest(
                device_id=TARGET_EDGE_ID,
                video_data=video_data
            )
    finally:
        cam_manager.stop_all()

async def main():
    print(f"🚀 가상 엣지 디바이스 ({TARGET_EDGE_ID}) 클라이언트 시작...")

    # [ONVIF 어댑터 초기설정] 엣지 장치의 로컬 네트워크에 존재하는 IP 카메라들을 브릿징 등록
    onvif_adapter = ONVIFAdapter()
    onvif_adapter.register_camera("cam_01", "10.1.2.10", 80, "admin", "byda13245")
    onvif_adapter.register_camera("cam_02", "192.168.0.102", 80, "admin", "1234")
    onvif_adapter.register_camera("cam_03", "192.168.0.103", 80, "admin", "1234")
    
    # 서버(edge_manager)의 50051 포트로 gRPC 연결
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = edge_communication_pb2_grpc.EdgeManagerStub(channel)
        
        try:
            # 양방향 스트리밍(RPC) 함수 호출
            call = stub.StreamDataAndControl(generate_requests())
            
            print(f"[{TARGET_EDGE_ID}] 서버에 연결 성공. 제어 명령을 기다립니다...\n")
            
            # 서버에서 내려오는 제어 명령(Downlink)을 계속 리스닝
            async for response in call:
                print("\n=======================================================")
                print(f"📥 [SERVER -> EDGE] 제어 명령 수신 (Downlink)!")
                print(f"   - Command: {response.command}")
                
                try:
                    # 파라미터가 비어있지 않으면 JSON으로 파싱 시도
                    params = json.loads(response.parameters) if response.parameters else {}
                    print(f"   - Parameters: {params}")
                    
                    # 어느 카메라(cam_id)에 명령할지 확인 (기본값 cam_01)
                    cam_id = params.get("camera_id", "cam_01")
                    
                    # ONVIF 어댑터를 통해 명령(PTZ/재부팅 등) 라우팅
                    onvif_adapter.handle_command(cam_id, response.command, params)
                    
                except json.JSONDecodeError:
                    print(f"   - Parameters(Raw): {response.parameters}")
                    print("     (파라미터 파싱 실패: 올바른 JSON 포맷이 아님)")
                
                print("=======================================================\n")
                
        except grpc.RpcError as e:
            print(f"\n[오류] gRPC 연결이 끊겼거나 접근할 수 없습니다: {e.details()}")
        except asyncio.CancelledError:
            print("\n클라이언트가 강제 종료되었습니다.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n사용자에 의해 종료되었습니다.")
