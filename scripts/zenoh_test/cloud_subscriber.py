# This file is part of DT_SERVER.
# 
# DT_SERVER is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
# 
# DT_SERVER is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DT_SERVER; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

import zenoh
import time
import json
import csv

results = []
pub_echo = None

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
        
        # 왕복 시간(RTT) 측정을 위해 엣지로 다시 에코(Echo) 전송
        # 주의: 1MB 전체를 다시 보내면 병목이 생기므로 파싱된 json_str(메타데이터)만 보냅니다.
        if pub_echo:
            pub_echo.put(json_str.encode('utf-8'))
            
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
        
        if pub_echo:
            pub_echo.put(raw_payload)
            
    except Exception as e:
        print(f"제어 파싱 에러: {e}")

def run_subscriber():
    # 서버(클라우드) 본체가 직접 10050 포트를 열고 엣지의 접속을 기다리도록(Listen) 설정합니다.
    conf = zenoh.Config()
    conf.insert_json5("listen/endpoints", '["tcp/0.0.0.0:10050"]')
    
    print("클라우드 서버에서 10050 포트를 개방하고 엣지의 접속을 기다리는 중...")
    session = zenoh.open(conf)
    
    global pub_echo
    pub_echo = session.declare_publisher("dt/cloud/echo", reliability=zenoh.Reliability.BEST_EFFORT)
    
    # 엣지와 동일한 QoS 설정으로 토픽 구독
    sub_tensor = session.declare_subscriber("dt/edge/tensor", tensor_handler)
    sub_control = session.declare_subscriber("dt/edge/control", control_handler)
    
    print("수신 및 에코(Echo) 대기 중... (종료 후 결과를 저장하려면 Ctrl+C 를 누르세요)\n")
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
