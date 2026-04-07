import asyncio
import json
import websockets
import omni.usd
from pxr import UsdGeom, Gf

# 스크립트 내에서 직접 USD 맵 로드 (명령어 인자보다 확실하게 보장)
usd_path = "/home/dojan/All/2025_SejongUniv_All.usd"
omni.usd.get_context().open_stage(usd_path)
print(f"Loading stage from {usd_path} ...")

"""
Meta Sejong Digital Twin - WebSocket Client for Isaac Sim

[실행 방법]
1. Isaac Sim의 내장 Python 환경에 websockets 라이브러리 설치가 필요합니다.
   (Isaac Sim 기본 파이썬: ./python.sh -m pip install websockets)
2. Isaac Sim 상단 메뉴 [Window] -> [Script Editor]를 엽니다.
3. 이 스크립트 코드 전체를 복사하여 Script Editor에 붙여넣고 실행(Run) 버튼을 누릅니다.
"""

# DT_SERVER의 sim_backend 웹소켓 주소 (Isaac Sim이 구동되는 환경의 IP로 변경 필요)
SIM_BACKEND_WS_URL = "ws://127.0.0.1:8004/ws/sim"
USD_PEOPLE_ROOT = "/World/MetaSejong_People"

class MetaSejongDigitalTwin:
    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.units_per_meter = UsdGeom.GetStageMetersPerUnit(self.stage)
        self._ensure_root_group()
        
    def _ensure_root_group(self):
        """디지털 트윈 상에 사람 객체들을 모아둘 루트 그룹(Xform) 생성"""
        root_prim = self.stage.GetPrimAtPath(USD_PEOPLE_ROOT)
        if not root_prim.IsValid():
            UsdGeom.Xform.Define(self.stage, USD_PEOPLE_ROOT)

    def update_person_transform(self, track_id: int, position: dict, pose_skeleton: list, mesh_params: dict):
        """수신받은 사람(ID)의 절대 좌표(World Coordinate)를 USD 씬에 적용합니다."""
        person_path = f"{USD_PEOPLE_ROOT}/Person_{track_id}"
        person_prim = self.stage.GetPrimAtPath(person_path)
        
        # 1. 씬에 해당 인물이 없다면 새 USD 객체(더미 캡슐 등)를 생성합니다.
        #    추후 이 부분을 실제 SMPL 3D Mesh 에셋 로드로 교체하시면 됩니다.
        if not person_prim.IsValid():
            capsule = UsdGeom.Capsule.Define(self.stage, person_path)
            # 메타 세종 씬의 단위(Meter, cm 등)에 맞게 높이와 반경 설정
            height = 1.7 / self.units_per_meter
            radius = 0.3 / self.units_per_meter
            
            capsule.GetHeightAttr().Set(height)
            capsule.GetRadiusAttr().Set(radius)
            capsule.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.5, 0.9)])  # 파란색 식별자
            
            # 변환 연산자(Translate) 부여
            xformable = UsdGeom.Xformable(capsule)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp()
            person_prim = self.stage.GetPrimAtPath(person_path)
            
        # 2. 위치 (Translate) 갱신
        xformable = UsdGeom.Xformable(person_prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
                
        if translate_op is not None:
            # Isaac Sim 환경의 스케일(Meter 혹은 cm) 비율에 맞춰 전송받은 절대 좌표(Meter 기준)를 배율 조정합니다.
            scale_factor = 1.0 / self.units_per_meter
            x = position.get("x", 0.0) * scale_factor
            y = position.get("y", 0.0) * scale_factor
            z = position.get("z", 0.0) * scale_factor
            
            # Gf.Vec3d를 통해 USD 내부 위치 강제 동기화 (Y-up / Z-up 에 따라 x,y,z 순서 보정이 필요할 수 있습니다.)
            translate_op.Set(Gf.Vec3d(x, y, z))

        # 3. 추가적인 뼈대/메쉬 파라미터 제어 (여기에 로직 추가 가능)
        # if pose_skeleton:
        #    update_skeleton_joints(person_prim, pose_skeleton)

async def ws_listener_task():
    """백그라운드에서 동작하는 Async WebSocket 수신 태스크"""
    dt_manager = MetaSejongDigitalTwin()
    
    print(f"[Meta Sejong] 디지털 트윈 서버 웹소켓({SIM_BACKEND_WS_URL})에 연결을 시도합니다...")
    try:
        async with websockets.connect(SIM_BACKEND_WS_URL) as ws:
            print("[Meta Sejong] 서버에 성공적으로 연결되었습니다. 실시간 수신 대기 중...")
            while True:
                message = await ws.recv()
                data = json.loads(message)
                
                # 수신 받은 구조화 JSON 이벤트 처리
                if data.get("action") == "UpdateTransforms":
                    objects = data.get("objects", [])
                    for obj in objects:
                        track_id = obj.get("track_id")
                        position = obj.get("position")
                        pose = obj.get("pose", [])
                        mesh = obj.get("mesh", {})
                        
                        if track_id is not None and position is not None:
                            # USD Transform 업데이트 호출
                            dt_manager.update_person_transform(track_id, position, pose, mesh)
                            
    except websockets.exceptions.ConnectionClosed:
        print("[Meta Sejong] 서버와의 연결이 중단되었습니다.")
    except Exception as e:
        print(f"[Meta Sejong] 웹소켓 에러 발생: {e}")

# Isaac Sim은 자체적인 Event Loop(asyncio)를 구동하고 있습니다.
# 백그라운드 코루틴으로 예약하여 메인 UI(Viewport)가 멈추지 않도록 동작시킵니다.
asyncio.ensure_future(ws_listener_task())
