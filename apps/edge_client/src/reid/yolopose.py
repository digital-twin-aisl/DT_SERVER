import os
import numpy as np
import cv2
from ultralytics import YOLO
from typing import Dict

# 모델 경로 설정
# base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# model_path = os.path.join(base_dir, "models", "yolo11n-pose.pt")
# model = YOLO(model_path).to("cuda")

# 사용할 keypoint index
SELECTED_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2p - x1p) * (y2p - y1p)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def extract_poses_from_frame(
        model, frame, edge_id=1, 
        cam_id=0, frame_num=0
        ) -> list[Dict]:
    # results = model(frame)
    results = model.predict(frame, device=0, verbose=False)
    original_frame = frame.copy()
    all_bboxes = []
    persons = []
    person_idx_global = 0

    h, w = original_frame.shape[:2]

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        for person_idx, person_kps in enumerate(keypoints):
            if person_idx >= len(boxes):
                continue

            # 유효 keypoint 확인
            valid_keypoints = [
                (i, kp) for i, kp in enumerate(person_kps)
                if i in SELECTED_KEYPOINTS and kp[0] > 0 and kp[1] > 0
            ]
            if len(valid_keypoints) < 12:
                continue

            # pose vector 생성
            pose_vec = []
            for i in SELECTED_KEYPOINTS:
                x, y = person_kps[i]
                pose_vec.append([x, y] if x > 0 and y > 0 else [-1, -1])
            pose_vec = np.array(pose_vec)

            x_min, y_min, x_max, y_max = map(int, boxes[person_idx])

            # [보안 1] 프레임 경계값 넘지 않도록 보정
            x_min = max(0, min(x_min, w - 1))
            x_max = max(0, min(x_max, w - 1))
            y_min = max(0, min(y_min, h - 1))
            y_max = max(0, min(y_max, h - 1))

            # [보안 2] 잘못된 bbox 방지
            if x_min >= x_max or y_min >= y_max:
                continue

            # [보안 3] 중복 제거
            if any(compute_iou((x_min, y_min, x_max, y_max), prev) > 0.05 for prev in all_bboxes):
                continue
            all_bboxes.append((x_min, y_min, x_max, y_max))

            crop_img = original_frame[y_min:y_max, x_min:x_max]

            # [보안 4] 빈 crop 방지
            if crop_img.size == 0 or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                continue

            # [보안 5] float → uint8 보정
            if crop_img.dtype != np.uint8:
                crop_img = crop_img.astype(np.uint8)

            persons.append({
                "edge_id": edge_id,
                "cam": cam_id,
                "frame": frame_num,
                "person_idx": person_idx_global,
                "keypoints": pose_vec,
                "bbox": [x_min, y_min, x_max, y_max],
                "crop": crop_img
            })

            person_idx_global += 1

    return persons