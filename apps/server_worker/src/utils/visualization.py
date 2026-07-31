import cv2
import numpy as np

LIMBS = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # 머리
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 몸통, 팔
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], # 다리
    [5, 11], [6, 12] # 몸통-다리 연결
]

# 관절 및 뼈대 색상
JOINT_COLOR = (0, 255, 0)  # Green
LIMB_COLOR = (0, 0, 255)   # Red

def project_3d_to_2d(points_3d, calibration_data, cam_id):
    """3D 포인트를 특정 카메라의 2D 평면에 투영합니다."""
    cam_params = calibration_data.cameras[f'cam_{cam_id}']
    R = cam_params['R']
    T = cam_params['T']
    K = cam_params['K']
    
    # distCoeffs가 없는 경우 0으로 채운 배열 사용
    distCoeffs = cam_params.get('distCoeffs', np.zeros((4, 1)))

    points_2d, _ = cv2.projectPoints(points_3d, R, T, K, distCoeffs)
    return points_2d.reshape(-1, 2)

def draw_poses_on_image(image, poses_3d, calibration_data, cam_id):
    """이미지 위에 3D 포즈를 2D로 투영하여 그립니다."""
    vis_image = image.copy()
    
    for pose_3d in poses_3d:
        # 3D 좌표를 (num_joints, 3) 형태로 변환
        pose_3d_np = np.array(pose_3d).reshape(-1, 3)
        
        # 2D로 투영
        points_2d = project_3d_to_2d(pose_3d_np, calibration_data, cam_id)

        # 관절 그리기
        for x, y in points_2d:
            cv2.circle(vis_image, (int(x), int(y)), 5, JOINT_COLOR, -1)

        # 뼈대 그리기
        for limb in LIMBS:
            p1_idx, p2_idx = limb
            if p1_idx < len(points_2d) and p2_idx < len(points_2d):
                p1 = tuple(map(int, points_2d[p1_idx]))
                p2 = tuple(map(int, points_2d[p2_idx]))
                cv2.line(vis_image, p1, p2, LIMB_COLOR, 2)
                
    return vis_image

def show_visualizations(images, poses_3d, calibration_data):
    """여러 카메라 뷰에 대한 시각화 결과를 창으로 보여줍니다."""
    if not poses_3d:
        # 포즈가 없으면 원본 이미지만 표시
        for i, img in enumerate(images):
            cv2.imshow(f'Camera {i+1}', img)
    else:
        # 각 카메라 뷰에 포즈를 그려서 표시
        for i, img in enumerate(images):
            vis_img = draw_poses_on_image(img, poses_3d, calibration_data, i + 1)
            cv2.imshow(f'Camera {i+1}', vis_img)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True