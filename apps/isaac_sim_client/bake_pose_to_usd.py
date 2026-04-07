import pickle
import numpy as np
import argparse
import sys
try:
    from pxr import Usd, UsdGeom, Vt, Sdf
except ImportError:
    print("Error: pxr module not found. Please run this script using Isaac Sim's Python environment (e.g., python.sh) or ensure usd-core is installed.")
    sys.exit(1)

# 포즈 연결 정보 (LIMBS)
LIMBS = [
    [0, 1],
    [0, 2],
    [0, 3],
    [3, 4],
    [4, 5],
    [0, 9],
    [9, 10],
    [10, 11],
    [2, 6],
    [2, 12],
    [6, 7],
    [7, 8],
    [12, 13],
    [13, 14],
]
# 선 생성을 위한 정점 수 계산 (각 limb당 2개의 점)
CURVE_VERTEX_COUNTS = [2] * len(LIMBS)

def create_pose_usd(pkl_path="result.pkl", usd_path="animated_poses.usd", fps=30):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {pkl_path} not found.")
        return

    # USD Stage 생성
    stage = Usd.Stage.CreateNew(usd_path)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(data) - 1)
    stage.SetTimeCodesPerSecond(fps)
    
    # Z축을 위로 설정 (Isaac Sim 기본값)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z) 

    root_path = "/Poses"
    root_prim = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root_prim) # 이 줄을 추가하여 Default Prim 지정

    # 최대 수용 가능한 사람 수 인스턴스 (예: 10명)
    MAX_PEOPLE = 10 
    people_curves = []

    for p in range(MAX_PEOPLE):
        curve_path = f"{root_path}/Person_{p}"
        curve = UsdGeom.BasisCurves.Define(stage, curve_path)
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        
        # 보기 좋은 선 두께와 색상 지정
        widths_attr = curve.CreateWidthsAttr()
        widths_attr.Set(Vt.FloatArray([5.0])) # 적절한 두께로 수정 가능
        
        geom_prim = curve.GetPrim()
        geom_prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
        geom_prim.GetAttribute("primvars:displayColor").Set(Vt.Vec3fArray([(0.0, 1.0, 0.0)])) # 녹색
        
        people_curves.append(curve)

    for frame_idx, frame_data in enumerate(data):
        # pkl 데이터 구조에 맞게 수정 필요
        # 예시 구조: 프레임 리스트 > 튜플/딕셔너리 > 예측된 3D Pose 배열
        # 만약 test_pose.py 결과라면 frame_data[0]가 preds_3d일 수 있습니다.
        if isinstance(frame_data, tuple) or isinstance(frame_data, list):
            preds = frame_data[0] # test_pose.py 의 preds_3d 반환값 참조
        elif isinstance(frame_data, dict):
            preds = frame_data.get('pred', frame_data.get('preds_3d', None))
        else:
            preds = frame_data

        time_code = Usd.TimeCode(frame_idx)
        
        for p in range(MAX_PEOPLE):
            curve = people_curves[p]
            
            # 예측 데이터에 해당 사람이 존재하는 경우
            if preds is not None and len(preds) > 0 and p < len(preds):
                # shape가 [num_person, num_joints, 4 or 5] 인지 [num_joints, 3] 인지에 따라 수정
                pose = preds[p]
                
                # 예측 신뢰도나 유효성을 체크 (예: confidence가 -1이면 무효)
                # 모델 출력 형식에 맞춰 아래 조건을 조정하세요
                is_valid = True
                if pose.shape[-1] >= 4 and isinstance(pose[0, 3], (int, float, np.number)):
                    if pose[0, 3] == -1:
                        is_valid = False
                
                if is_valid:
                    xyz = pose[:, :3]
                    
                    # LIMBS 순서에 맞춰 포인트 배열 생성
                    points = []
                    for j1, j2 in LIMBS:
                        points.append(tuple(xyz[j1].tolist()))
                        points.append(tuple(xyz[j2].tolist()))
                    
                    # 프레임별 위치 및 정점 수 기록
                    curve.CreatePointsAttr().Set(Vt.Vec3fArray(points), time=time_code)
                    curve.CreateCurveVertexCountsAttr().Set(Vt.IntArray(CURVE_VERTEX_COUNTS), time=time_code)
                    
                    # 현재 프레임에서 보이게 설정
                    curve.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited, time=time_code)
                    continue

            # 데이터가 유효하지 않거나 없으면 이 프레임에서 숨김
            curve.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible, time=time_code)

    stage.GetRootLayer().Save()
    print(f"[{usd_path}] 생성이 완료되었습니다!")
    print("이제 Isaac Sim을 열고 해당 USD 파일을 Stage로 드래그 앤 드롭 하세요.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pkl pose results to interactive USD animation")
    parser.add_argument("--pkl", type=str, default="result.pkl", help="Path to the input result.pkl")
    parser.add_argument("--usd", type=str, default="animated_poses.usd", help="Path to save the output USD")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()
    
    create_pose_usd(args.pkl, args.usd, args.fps)
