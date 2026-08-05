import numpy as np

SELECTED_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2p - x1p) * max(0, y2p - y1p)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def extract_poses_from_frame(
    model, frame, edge_id=1, cam_id=0, frame_num=0
) -> list[dict]:
    results = model.predict(frame, device=0, verbose=False)
    frame = frame.copy()
    accepted_boxes = []
    persons = []
    height, width = frame.shape[:2]

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        for person_kps, box in zip(keypoints, boxes):
            pose = person_kps[SELECTED_KEYPOINTS]
            if not np.all(pose > 0):
                continue

            x1, y1, x2, y2 = np.asarray(box, dtype=int)
            bbox = (
                int(np.clip(x1, 0, width - 1)),
                int(np.clip(y1, 0, height - 1)),
                int(np.clip(x2, 0, width)),
                int(np.clip(y2, 0, height)),
            )
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            if any(compute_iou(bbox, previous) > 0.05 for previous in accepted_boxes):
                continue

            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if not crop.size:
                continue
            accepted_boxes.append(bbox)
            persons.append(
                {
                    "edge_id": edge_id,
                    "cam": cam_id,
                    "frame": frame_num,
                    "person_idx": len(persons),
                    "keypoints": pose,
                    "bbox": list(bbox),
                    "crop": crop.astype(np.uint8, copy=False),
                }
            )

    return persons
