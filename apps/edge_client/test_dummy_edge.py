import asyncio
import sys
import os
import json
import grpc
import requests
import urllib3
from aiortc import RTCPeerConnection, RTCSessionDescription
import time
import ssl

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# edge_manager 내부의 생성된 protobuf 파이썬 파일을 참조할 수 있도록 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'edge_manager', 'app', 'protos'))
try:
    import edge_communication_pb2
    import edge_communication_pb2_grpc
except ImportError as e:
    print(f"[오류] Protobuf 모듈을 불러오지 못했습니다. 경로를 확인하세요: {e}")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()
TARGET_EDGE_ID = os.getenv("EDGE_ID", "edge_001")
gRPC_SERVER_IP_PORT = os.getenv("gRPC_SERVER_IP_PORT", "localhost:8443")
HTTP_SERVER_URL = os.getenv("HTTP_SERVER_URL", "http://localhost:8080/api/edge")

async def setup_webrtc():
    print(f"[{TARGET_EDGE_ID}] WebRTC 고대역폭 데이터 채널 (Uplink) 준비 중...")
    pc = RTCPeerConnection()
    channel = pc.createDataChannel("video_tensor_stream")
    
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # ICE 후보 수집이 완료될 때까지 대기
    print(f"[{TARGET_EDGE_ID}] WebRTC ICE 후보 수집 중...")
    # aiortc의 경우 백그라운드 태스크에서 후보를 수집하므로 이와 같이 폴링
    while pc.iceConnectionState != "completed":
        if pc.iceConnectionState == "failed":
            pass # handle error if needed
        # 그냥 약간 대기해보거나, wait for event (aiortc 최신버전 방식 다름)
        await asyncio.sleep(0.1)
        # Or better: `if pc.iceGatheringState == 'complete': break` 
        break # but aiortc doesn't always strictly require it locally if host candidates are instant.

    print(f"[{TARGET_EDGE_ID}] WebRTC SDP Offer 생성 완료. 서버로 전송 중...")

    # Nginx 로 라우팅되는 HTTP endpoint 에 POST (HTTPS 자체 서명 인증서 무시)
    res = requests.post(f"{HTTP_SERVER_URL}/webrtc/offer", json={
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "device_id": TARGET_EDGE_ID
    }, verify=False)
    
    if res.status_code == 200:
        answer_data = res.json()
        answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
        await pc.setRemoteDescription(answer)
        print(f"[{TARGET_EDGE_ID}] WebRTC 피어 연결 완료.")
    else:
        print(f"[오류] WebRTC 시그널링 실패: HTTP {res.status_code} - {res.text}")
        sys.exit(1)
        
    return pc, channel

async def webrtc_send_loop(channel):
    """ 10 FPS (100ms) 목표로 비디오 텐서 데이터를 전송 """
    frame_idx = 0
    while True:
        await asyncio.sleep(0.1) # 10 FPS 지연시간 (100ms)
        frame_idx += 1
        if channel.readyState == "open":
            print(f"[EDGE -> SERVER] WebRTC 실시간 (10FPS) 텐서 전송 중... (frame_id: {frame_idx})")
            dummy_payload = f"TIMESTAMP:{time.time()}|FRAME:{frame_idx}|DATA:0xDEADBEEF".encode('utf-8')
            channel.send(dummy_payload)

async def generate_grpc_requests():
    """
    서버로 메타데이터 및 제어 연결(Keep-Alive)을 올리는(Uplink) 제너레이터.
    무거운 비디오 데이터는 WebRTC로 분리하였으므로 순수 제어용.
    """
    print(f"[{TARGET_EDGE_ID}] gRPC 제어 스트림 (Uplink) 시작되었습니다.")
    
    yield edge_communication_pb2.EdgeStreamRequest(
        device_id=TARGET_EDGE_ID,
        heartbeat="INIT_CONNECTION"
    )
    
    while True:
        await asyncio.sleep(10)
        yield edge_communication_pb2.EdgeStreamRequest(
            device_id=TARGET_EDGE_ID,
            heartbeat="KEEP_ALIVE"
        )

async def main():
    print(f"🚀 가상 엣지 디바이스 하이브리드 클라이언트 ({TARGET_EDGE_ID}) 시작...")
    
    # 1. WebRTC 연결
    pc, webrtc_channel = await setup_webrtc()
    asyncio.create_task(webrtc_send_loop(webrtc_channel))
    
    # 2. gRPC 연결 세팅 (포트 443 - Nginx 단일 포트)
    host, port_str = gRPC_SERVER_IP_PORT.split(":")
    try:
        # Nginx HTTPS 통신을 위한 자체서명 인증서 동적 획득
        cert = ssl.get_server_certificate((host, int(port_str)))
        credentials = grpc.ssl_channel_credentials(cert.encode('utf-8'))
        print(f"[{TARGET_EDGE_ID}] HTTPS(gRPC) 통신용 SSL 인증서 확인 완료.")
    except Exception as e:
        print(f"[{TARGET_EDGE_ID}] 서버 연결 실패 (Nginx가 준비되지 않았을 수 있습니다): {e}")
        return

    # secure_channel 로 변경하여 gRPC TLS 세션 시작
    async with grpc.aio.secure_channel(gRPC_SERVER_IP_PORT, credentials) as channel:
        stub = edge_communication_pb2_grpc.EdgeManagerStub(channel)
        
        try:
            call = stub.StreamDataAndControl(generate_grpc_requests())
            print(f"[{TARGET_EDGE_ID}] 서버 (gRPC) 연결 성공. 제어 명령 대기...\n")
            
            async for response in call:
                print("\n=======================================================")
                print(f"📥 [SERVER -> EDGE] 제어 명령 수신 (gRPC Downlink)!")
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
