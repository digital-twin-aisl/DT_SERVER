from fastapi import FastAPI
import redis.asyncio as redis
import os
import json

app = FastAPI(title="Camera Manager Service")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

@app.on_event("startup")
async def startup_event():
    pass

@app.get("/")
def read_root():
    return {"message": "Camera Manager is running."}

@app.get("/api/cameras/{edge_id}/{camera_id}/calibration")
async def get_camera_calibration(edge_id: str, camera_id: int):
    """
    특정 엣지 디바이스의 카메라 캘리브레이션 파라미터를 조회합니다.
    """
    # 실제로는 PostgreSQL 등 DB에서 조회하지만 여기서는 하드코딩 값 반환
    return {
        "x": 3.0, 
        "y": 1.5,
        "z": 2.8,
        "pitch": 0.0, 
        "yaw": 0.0, 
        "roll": 0.0
    }

@app.post("/api/calibration/request")
async def request_calibration_data(edge_id: str, camera_id: int):
    """
    엣지 디바이스에 특정 카메라의 캘리브레이션용 이미지/특징 데이터를 요구합니다.
    """
    command_payload = {
        "command": "REQUEST_CALIBRATION_DATA",
        "target_camera_id": camera_id
    }
    
    # edge_manager가 구독하고 엣지로 전달할 수 있도록 Redis Pub/Sub 사용
    await redis_client.publish(f"control:edge:{edge_id}", json.dumps(command_payload))
    
    return {"status": "success", "message": f"Calibration request sent to edge {edge_id} for camera {camera_id}"}
