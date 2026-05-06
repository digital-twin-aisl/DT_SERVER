import zenoh
import time
import json
import numpy as np
import sys

# 기본 클라우드 접속 주소 (서버 IP로 변경 가능)
CLOUD_ZENOH_IP = "127.0.0.1" 
CLOUD_ZENOH_PORT = 7447

def run_publisher():
    # 1. 클라우드의 Zenoh 라우터에 연결 설정
    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", f'["tcp/{CLOUD_ZENOH_IP}:{CLOUD_ZENOH_PORT}"]')
    
    print(f"[{CLOUD_ZENOH_IP}] 클라우드 Zenoh 라우터에 연결 중...")
    session = zenoh.open(conf)
    
    # 2. 두 가지 QoS(Quality of Service) 퍼블리셔 생성
    # - Best Effort: 텐서 데이터 (1MB, 초당 10번). 유실 허용, 최저 지연 우선. (UDP 특성)
    pub_tensor = session.declare_publisher("dt/edge/tensor", reliability=zenoh.Reliability.BEST_EFFORT)
    
    # - Reliable: 제어 데이터 (작은 크기). 유실 불가, 재전송 보장. (TCP 특성)
    pub_control = session.declare_publisher("dt/edge/control", reliability=zenoh.Reliability.RELIABLE)
    
    print("퍼블리셔 생성 완료! 통신 테스트를 시작합니다...\n")
    
    # 테스트 환경 파라미터
    fps = 10
    duration = 10 # 10초간 전송
    total_frames = fps * duration
    interval = 1.0 / fps
    
    print(f"--- 1. [Best Effort] 무거운 텐서 데이터(1MB) {fps}FPS 전송 시작 ---")
    for i in range(total_frames):
        timestamp = time.time()
        payload = {
            "frame_id": i,
            "timestamp": timestamp,
        }
        # JSON 메타데이터 뒤에 1MB 크기가 되도록 더미 0바이트를 패딩하여 무거운 텐서를 흉내냄
        json_bytes = json.dumps(payload).encode('utf-8')
        msg = json_bytes.ljust(1024 * 1024, b'0')
        
        pub_tensor.put(msg)
        print(f"[Tensor Best-Effort] Frame {i} 전송 완료 (크기: 1MB)")
        time.sleep(interval)
        
    time.sleep(2) # 채널 변경 대기
        
    print("\n--- 2. [Reliable] 중요한 제어 명령 데이터 전송 시작 ---")
    for i in range(5):
        timestamp = time.time()
        payload = {
            "cmd_id": i,
            "timestamp": timestamp,
            "command": "TRIGGER_CALIBRATION"
        }
        msg = json.dumps(payload).encode('utf-8')
        
        pub_control.put(msg)
        print(f"[Control Reliable] Command {i} 전송 완료")
        time.sleep(0.5)
        
    session.close()
    print("\n엣지 퍼블리셔 테스트가 정상적으로 종료되었습니다.")

if __name__ == "__main__":
    # 실행 시 인자로 서버 IP를 넘겨받을 수 있습니다. 예: python edge_publisher.py 210.xx.xx.xx
    if len(sys.argv) > 1:
        CLOUD_ZENOH_IP = sys.argv[1]
    run_publisher()
