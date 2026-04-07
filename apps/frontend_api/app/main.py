import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib

app = FastAPI(title="Frontend API Gateway")

# 프론트엔드(React, Vue 등) 브라우저에서 직접 API를 찌를 수 있도록 CORS 허용 세팅
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 배포 환경에서는 실제 도메인으로 대체 요망
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 웹 브라우저 제공 설정
static_dir = pathlib.Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 내부 Docker 컨테이너 주소 (docker-compose의 서비스명 기준, 기본 컨테이너 포트 80 사용)
CAMERA_MANAGER_URL = os.getenv("CAMERA_MANAGER_URL", "http://camera_manager:80")
EDGE_MANAGER_URL = os.getenv("EDGE_MANAGER_URL", "http://edge_manager:80")

@app.get("/")
def read_root():
    # 웰컴 페이지 대신 메인 대시보드(SPA) 제공
    index_file = pathlib.Path(__file__).parent / "static" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Frontend API Gateway is up. Static dashboard not found.",
        "docs": "Access /docs for Swagger UI"
    }

@app.get("/api/v1/system/status")
async def get_system_status():
    """간단한 시스템 헬스체크 및 통합 상태 정보"""
    # 실제로는 Redis에 각 모듈의 상태를 핑 쳐서 취합하거나, 각 모듈의 루트("/")를 찔러서 확인합니다.
    return {
        "edge_manager": "online",
        "dl_worker": "online",
        "camera_manager": "online",
        "sim_backend": "online"
    }

@app.get("/api/v1/edges")
async def get_edges_list():
    """Camera Manager로부터 엣지 및 카메라 메타데이터 데이터베이스 목록을 가져옵니다."""
    async with httpx.AsyncClient() as client:
        try:
            # camera_manager 컨테이너의 내부망 포트 80으로 요청
            response = await client.get(f"{CAMERA_MANAGER_URL}/api/edges", timeout=5.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Camera Manager 연동 에러: {e}")

@app.post("/api/v1/control/calibration")
async def request_calibration_trigger(edge_id: str, camera_id: int):
    """
    Frontend 사용자가 '캘리브레이션 시작' 버튼을 눌렀을 때, 
    요청을 일관된 포멧으로 래핑하여 Camera Manager 쪽으로 푸시 (Proxy) 합니다.
    """
    async with httpx.AsyncClient() as client:
        try:
            # camera_manager 컨테이너로 파라미터 전달 및 Proxy 호출
            response = await client.post(
                f"{CAMERA_MANAGER_URL}/api/calibration/request",
                params={"edge_id": edge_id, "camera_id": camera_id},
                timeout=5.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Camera Manager 제어 에러: {e}")

