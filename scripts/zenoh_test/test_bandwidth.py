import zenoh
import time
import threading
import os

# 1. 클라우드 수신부 (라우터 역할)
def cloud_node():
    conf = zenoh.Config()
    conf.insert_json5("listen/endpoints", '["tcp/127.0.0.1:10050"]')
    session = zenoh.open(conf)
    print("[Cloud] Listening on tcp/127.0.0.1:10050...")
    
    def handler(sample):
        pass # 그냥 받기만 함
    sub = session.declare_subscriber("dt/edge/tensor", handler)
    time.sleep(10)
    session.close()

# 2. 다른 엣지 (아무것도 구독하지 않거나 다른 토픽 구독)
def dummy_edge():
    time.sleep(1) # 라우터가 켜질 때까지 대기
    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", '["tcp/127.0.0.1:10050"]')
    session = zenoh.open(conf)
    print("[Dummy Edge] Connected to Cloud. Subscribing to 'dt/edge/other'...")
    
    def handler(sample):
        print(f"[Dummy Edge] Received data! (This shouldn't happen)")
    sub = session.declare_subscriber("dt/edge/other", handler)
    
    time.sleep(8)
    print("[Dummy Edge] Test finished. Closing session.")
    session.close()

# 3. 송신 엣지 (무거운 데이터 송신)
def publisher_edge():
    time.sleep(2) # 라우터와 Dummy가 연결될 때까지 대기
    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", '["tcp/127.0.0.1:10050"]')
    session = zenoh.open(conf)
    pub = session.declare_publisher("dt/edge/tensor")
    
    print("[Publisher Edge] Connected. Sending 5MB of data to 'dt/edge/tensor'...")
    large_payload = b'0' * (5 * 1024 * 1024) # 5MB 데이터
    pub.put(large_payload)
    print("[Publisher Edge] 5MB data sent successfully.")
    
    time.sleep(2)
    session.close()

if __name__ == "__main__":
    t1 = threading.Thread(target=cloud_node)
    t2 = threading.Thread(target=dummy_edge)
    t3 = threading.Thread(target=publisher_edge)
    
    t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()
    print("All nodes closed.")
