import asyncio
import os
import json
from fastapi import FastAPI
import httpx
import redis.asyncio as redis

from app.protos import edge_communication_pb2
from app.core.inference import DLInferencer

app = FastAPI(title="Dl Worker Service")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=False)
CAMERA_MANAGER_URL = os.getenv("CAMERA_MANAGER_URL", "http://camera_manager:80")

# 싱글톤 모델 로딩
inferencer = DLInferencer()

async def get_camera_calibration(edge_id: str, camera_id: int) -> dict:
    """
    camera_manager에서 DB에 있는 캘리브레이션 획득. 반환값을 Redis 캐싱하여
    비디오 프레임 단위의 병목을 막음
    """
    cache_key = f"calib:edge:{edge_id}:cam:{camera_id}"
    cached_val = await redis_client.get(cache_key)
    
    if cached_val:
        return json.loads(cached_val.decode('utf-8'))
        
    print(f"[{cache_key}] 캘리브레이션 정보를 camera_manager 에서 획득(동기화) 중...")
    
    # httpx를 통해 camera_manager/api/cameras/... 호출
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{CAMERA_MANAGER_URL}/api/cameras/{edge_id}/{camera_id}/calibration",
                timeout=5.0
            )
            resp.raise_for_status()
            calibration_data = resp.json()
    except Exception as e:
        print(f"[경고] camera_manager 호출 실패: {e}. 기본 캘리브레이션 사용.")
        calibration_data = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0
        }
    
    await redis_client.set(cache_key, json.dumps(calibration_data).encode('utf-8'), ex=600) # 10분 캐시
    return calibration_data

async def consume_video_stream():
    """
    Redis Stream에서 실시간으로 video_data(피처맵, 복셀)를 꺼내어
    딥러닝 연산 및 절대 좌표 변환을 수행합니다.
    """
    print("========== [DL Worker] Redis Stream 컨슈머 시작 ==========")
    last_id = b"$"
    
    while True:
        try:
            messages = await redis_client.xread(
                streams={"stream:video_data": last_id},
                count=10, block=100 
            )
            
            if messages:
                for stream_name, _messages in messages:
                    for message_id, data in _messages:
                        device_id = data.get(b"device_id").decode('utf-8')
                        payload_bytes = data.get(b"payload")
                        
                        video_data = edge_communication_pb2.VideoFeatureData()
                        video_data.ParseFromString(payload_bytes)
                        
                        # 1. 대상 카메라 캘리브레이션 획득 (캐시 기반)
                        calib_params = await get_camera_calibration(device_id, video_data.camera_id)
                        
                        # 2. 딥러닝 추론 (상대 3D 추출 후 절대 세계 좌표 변환)
                        inference_result = inferencer.process_tensor(
                            feature_map=video_data.feature_map, 
                            voxel_data=video_data.voxel_data,
                            camera_id=video_data.camera_id,
                            calib_params=calib_params
                        )
                        
                        inference_result["device_id"] = device_id
                        inference_result["camera_id"] = video_data.camera_id
                        inference_result["frame_id"] = video_data.frame_id
                        inference_result["timestamp"] = video_data.timestamp
                        
                        print(f"[DL Worker] 추론 및 좌표 변환 완료 [{device_id}:{video_data.camera_id}] Frame: {video_data.frame_id}")
                        
                        # 3. Isaac 연동 pubsub 통신
                        await redis_client.publish(
                            "pubsub:inference_result", 
                            json.dumps(inference_result).encode('utf-8')
                        )
                        
                        last_id = message_id

        except asyncio.CancelledError:
            print("컨슈머 루프 종료")
            break
        except Exception as e:
            print(f"Redis 읽기 중 예외 발생: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    # 서버 기동 시 컨슈머 태스크를 백그라운드에 등록
    asyncio.create_task(consume_video_stream())

@app.get("/")
def read_root():
    return {"message": "Hello from dl_worker. AI Consumer is running."}
