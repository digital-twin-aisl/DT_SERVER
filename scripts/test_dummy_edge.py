import asyncio
import sys
import os
import grpc

# edge_manager 내부의 생성된 protobuf 파이썬 파일을 참조할 수 있도록 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'edge_manager', 'app', 'protos'))
try:
    import edge_communication_pb2
    import edge_communication_pb2_grpc
except ImportError as e:
    print(f"[오류] Protobuf 모듈을 불러오지 못했습니다. 경로를 확인하세요: {e}")
    sys.exit(1)

TARGET_EDGE_ID = "edge_001"

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
    frame_idx = 0
    while True:
        await asyncio.sleep(5)
        frame_idx += 1
        
        print(f"[EDGE -> SERVER] 가상 비디오 프레임 전송 중... (frame_id: {frame_idx})")
        
        video_data = edge_communication_pb2.VideoFeatureData(
            timestamp=1234567890,
            frame_id=str(frame_idx),
            camera_id=1, feature_map=b"dummy", voxel_data=b"dummy"
        )
        
        yield edge_communication_pb2.EdgeStreamRequest(
            device_id=TARGET_EDGE_ID,
            video_data=video_data
        )

async def main():
    print(f"🚀 가상 엣지 디바이스 ({TARGET_EDGE_ID}) 클라이언트 시작...")
    
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
                print(f"   - Parameters: {response.parameters}")
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
