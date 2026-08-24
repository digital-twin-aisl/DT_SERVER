import glob
import cv2
import os.path as osp
import pickle
import threading

import numpy as np

from dt_common.calibration.voxelpose import (
    load_calibration_result,
    select_edge_config_voxelpose_cameras,
    select_voxelpose_cameras,
)
from apps.edge_client.src.utils.transforms import get_scale
from apps.edge_client.src.utils.input import (
    discover_dataset_videos,
    find_dataset_calibration,
)


class CalibrationData:
    '''
    멀티 스레딩 환경에서 RTSP 카메라 캘리브레이션 데이터에 접근하고 업데이트를 용이하게 하기 위한 클래스
        calibration = CalibrationData(config.CAMERA)
        calib_data = calibration.get() # 항상 최신 값 사용
        calibration.update(config.CAMERA) # 캘리브레이션 값을 업데이트
    
    미래 수정을 위한 selfpose3d dataset 코드
    M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]) # OpenCV 좌표계(오른손 좌표계)
    R, _ = cv2.Rodrigues(calib['rvec'])
    # R = R.dot(M)
    T = (
        -np.dot(R.T, calib['tvec']) * 1000 # mm단위 변환
    )
    '''
    def __init__(
        self,
        cfg,
        example_path=None,
        cameras=None,
        calibration_path=None,
        camera_ids=None,
        world_origin_m=(0.0, 0.0, 0.0),
    ):
        self._lock = threading.Lock()
        self.rtsp_cam = cameras
        self.cams = []
        self.example_path = example_path
        self.camera_ids = None if camera_ids is None else list(camera_ids)
        self.world_origin_m = tuple(world_origin_m)

        self.orig_image_size = np.array(cfg.NETWORK.IMAGE_SIZE_ORIG)
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.c = np.array(
            [self.orig_image_size[0] / 2.0, self.orig_image_size[1] / 2.0]
        )
        self.s = get_scale(self.orig_image_size, self.image_size)
        self.r = 0

        if calibration_path is not None:
            result = load_calibration_result(calibration_path)
            self.cams = select_voxelpose_cameras(
                result,
                camera_ids,
                world_origin_m=world_origin_m,
            )
        elif self.example_path:
            self.update_from_dataset()
        elif self.rtsp_cam:
            self.update()

        cfg.CAMS = self.cams

    def get(self):
        with self._lock:
            return self.cams

    def update(self):
        camera_ids = self.camera_ids or [camera["id"] for camera in self.rtsp_cam]
        self.cams = select_edge_config_voxelpose_cameras(
            self.rtsp_cam,
            camera_ids,
            world_origin_m=self.world_origin_m,
        )

    def update_from_dataset(self):
        result_path = find_dataset_calibration(self.example_path)
        if result_path is not None:
            camera_ids = self.camera_ids or [
                camera_id
                for camera_id, _ in discover_dataset_videos(self.example_path)
            ]
            result = load_calibration_result(result_path)
            self.cams = select_voxelpose_cameras(result, camera_ids)
            return

        cams = []
        calibration_paths = sorted(
            glob.glob(osp.join(self.example_path, "calibration", "*.pkl"))
        )
        for path in calibration_paths:
            with open(path, "rb") as f:
                calib = pickle.load(f)
                
            R, _ = cv2.Rodrigues(calib['rvec'])
            T = (
                -np.dot(R.T, calib['tvec']) * 1000
            )

            # m 딕셔너리 생성
            cam = {
                'R': R,
                'T': T,  
                'fx': calib['camera_matrix'][0, 0],
                'fy': calib['camera_matrix'][1, 1],
                'cx': calib['camera_matrix'][0, 2],
                'cy': calib['camera_matrix'][1, 2],
                'k': calib['dist_coeffs'][0][[0,1,4]].reshape(3, 1),  # 왜곡 계수 k1, k2, k3
                'p': calib['dist_coeffs'][0][[2,3]].reshape(2, 1)  # 왜곡 계수 p1, p2
            }
            cams.append(cam)
        self.cams = cams
