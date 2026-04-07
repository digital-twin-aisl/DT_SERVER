import asyncio
import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import redis.asyncio as redis

app = FastAPI(title="Sim Backend Service")

# Redis 연결 설정
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
redis_pubsub = redis_client.pubsub()

# WebSocket 연결을 관리하는 Connection Manager 클래스
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[Sim Backend] 클라이언트 접속 (현재 접속자: {len(self.active_connections)})")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"[Sim Backend] 클라이언트 퇴장 (현재 접속자: {len(self.active_connections)})")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"[Sim Backend] 메시지 브로드캐스트 오류: {e}")

manager = ConnectionManager()

async def subscribe_inference_results():
    """
    백그라운드에서 Redis Pub/Sub 채널 'pubsub:inference_result'를 리스닝하며,
    추론 데이터가 들어올 때마다 접속된 모든 WebSocket 클라이언트(Isaac Sim 등)에게 뿌려줍니다.
    """
    print("========== [Sim Backend] Redis 추론 결과 리스너(Subscriber) 시작 ==========")
    await redis_pubsub.subscribe("pubsub:inference_result")
    
    try:
        async for message in redis_pubsub.listen():
            if message["type"] == "message":
                raw_data = message["data"]
                
                try:
                    parsed = json.loads(raw_data)
                    
                    # Isaac Sim을 위한 명확한 이벤트/스키마 포맷 구조화
                    sim_payload = {
                        "action": "UpdateTransforms",
                        "metadata": {
                            "device_id": parsed.get("device_id"),
                            "camera_id": parsed.get("camera_id"),
                            "frame_id": parsed.get("frame_id"),
                            "timestamp": parsed.get("timestamp")
                        },
                        "objects": parsed.get("persons", [])
                    }
                    
                    # 클라이언트(Isaac Sim, Frontend GUI)에게 3D 렌더링용 구조화된 JSON 스트링 릴레이
                    await manager.broadcast(json.dumps(sim_payload))
                except json.JSONDecodeError:
                    print(f"[Sim Backend] JSON 파싱 오류: {raw_data}")
                    
    except asyncio.CancelledError:
        print("========== [Sim Backend] 추론 결과 리스너 중지 (Cancelled) ==========")
    except Exception as e:
        print(f"========== [Sim Backend] 추론 결과 리스너 에러: {e} ==========")

@app.on_event("startup")
async def startup_event():
    # 서버 기동 시 Redis 결과 리스너를 백그라운드 태스크로 구동
    asyncio.create_task(subscribe_inference_results())

@app.get("/")
def read_root():
    return {"message": "Hello from Sim Backend. WebSockets are ready."}

@app.websocket("/ws/sim")
async def websocket_endpoint(websocket: WebSocket):
    """
    Isaac Sim 또는 클라이언트 3D 뷰어가 이 주소로 WebSocket 연결을 맺어
    동적(Dynamic) 객체 변동 사항들을 실시간으로 제공받습니다.
    (예: ws://localhost:8004/ws/sim)
    """
    await manager.connect(websocket)
    try:
        while True:
            # 여기서는 서버가 클라이언트 메시지를 대기(ping/명령 등)하도록 할 수 있습니다.
            # 지금은 수신보다는 송신(Broadcast) 목적이 더 큽니다.
            data = await websocket.receive_text()
            print(f"[Sim Backend] 클라이언트로부터 온 메시지: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
