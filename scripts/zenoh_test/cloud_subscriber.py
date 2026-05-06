import zenoh
import time
import json
import csv

results = []

def tensor_handler(sample):
    recv_time = time.time()
    raw_payload = sample.payload.to_bytes()
    try:
        # 패딩된 더미 바이트에서 실제 JSON 부분만 파싱
        end_idx = raw_payload.find(b'}') + 1
        json_str = raw_payload[:end_idx].decode('utf-8')
        data = json.loads(json_str)
        
        latency = (recv_time - data['timestamp']) * 1000 # 밀리초(ms) 변환
        results.append({
            "type": "tensor_best_effort",
            "id": data['frame_id'],
            "send_time": data['timestamp'],
            "recv_time": recv_time,
            "latency_ms": latency
        })
        print(f"[클라우드 수신] 텐서 프레임 {data['frame_id']:03d} | E2E 지연시간: {latency:.2f} ms")
    except Exception as e:
        print(f"텐서 파싱 에러: {e}")

def control_handler(sample):
    recv_time = time.time()
    raw_payload = sample.payload.to_bytes()
    try:
        data = json.loads(raw_payload.decode('utf-8'))
        latency = (recv_time - data['timestamp']) * 1000
        results.append({
            "type": "control_reliable",
            "id": data['cmd_id'],
            "send_time": data['timestamp'],
            "recv_time": recv_time,
            "latency_ms": latency
        })
        print(f"[클라우드 수신] 제어 명령 {data['cmd_id']:03d} | E2E 지연시간: {latency:.2f} ms | 명령: {data['command']}")
    except Exception as e:
        print(f"제어 파싱 에러: {e}")

def run_subscriber():
    # 서버는 이미 docker-compose로 zenoh-bridge-dds가 떠 있으므로 로컬 라우터에 연결합니다.
    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", '["tcp/127.0.0.1:7447"]')
    
    print("로컬 Zenoh 라우터(127.0.0.1:7447)에 접속 중...")
    session = zenoh.open(conf)
    
    # 엣지와 동일한 QoS 설정으로 토픽 구독
    sub_tensor = session.declare_subscriber("dt/edge/tensor", tensor_handler, reliability=zenoh.Reliability.BEST_EFFORT())
    sub_control = session.declare_subscriber("dt/edge/control", control_handler, reliability=zenoh.Reliability.RELIABLE())
    
    print("수신 대기 중... (종료 후 결과를 저장하려면 Ctrl+C 를 누르세요)\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n수신을 종료하고 지연시간(Latency) 측정 결과를 저장합니다...")
    finally:
        session.close()
        
        # CSV 파일로 실험 결과 저장
        csv_filename = 'zenoh_test_results.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["type", "id", "send_time", "recv_time", "latency_ms"])
            writer.writeheader()
            writer.writerows(results)
        print(f"결과 저장 완료: {csv_filename}")
        print("시각화를 위해 python visualize.py 를 실행하세요.")

if __name__ == "__main__":
    run_subscriber()
