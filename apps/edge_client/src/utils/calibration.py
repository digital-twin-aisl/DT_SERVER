import json
import threading
import os.path as osp
import glob
import cv2
import pickle
import numpy as np

from apps.edge_client.src.utils.transforms import get_scale

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
    def __init__(self, cfg, example_path=None, cameras=None):
        self._lock = threading.Lock()
        self.rtsp_cam = cameras
        self.cams = []
        self.example_path = example_path

        self.orig_image_size= np.array(cfg.NETWORK.IMAGE_SIZE_ORIG)
        self.image_size=np.array(cfg.NETWORK.IMAGE_SIZE)
        self.c = np.array([self.orig_image_size[0] / 2.0, self.orig_image_size[1] / 2.0])
        self.s = get_scale(self.orig_image_size, self.image_size)
        self.r = 0

        if self.example_path:
            self.update_from_dataset()
        elif self.rtsp_cam: 
            self.update()

        cfg.CAMS = self.cams


        
        
    def get(self):
        with self._lock:
            return self.cams
        
    def update(self):
        self.camera_calibration_paths = osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "camera_calibration_results.json")
        with open(self.camera_calibration_paths, "r") as f:
            all_results = json.load(f)
        for camera_source in self.rtsp_cam:
            # New edge-local registrations use a stable logical camera ID so
            # calibration keys never contain a physical endpoint or password.
            cam_id_str = str(camera_source['id'])
            legacy_id = "".join(
                c for c in str(camera_source['url']) if c.isalnum() or c in '_-'
            )
            result_key = cam_id_str if cam_id_str in all_results else legacy_id
            if result_key not in all_results:
                raise KeyError(f"카메라 ID '{cam_id_str}'에 대한 캘리브레이션 결과를 찾을 수 없습니다.")
            
            result = all_results[result_key]
            R, _ = cv2.Rodrigues(np.array(result["rvec"]))
            T = (
                -np.dot(R.T, np.array(result["tvec"])) * 1000  # mm 단위 변환
            )
            cam = {
                'id': camera_source['id'],
                'R': R,
                'T': T,
                'fx': result["camera_matrix"][0][0],
                'fy': result["camera_matrix"][1][1],
                'cx': result["camera_matrix"][0][2],
                'cy': result["camera_matrix"][1][2],
                'k': np.array(result["dist_coeffs"])[[0, 1, 4]].reshape(3, 1),  # 왜곡 계수 k1, k2, k3
                'p': np.array(result["dist_coeffs"])[[2, 3]].reshape(2, 1)  # 왜곡 계수 p1, p2
            }
            self.cams.append(cam)
            print(f"카메라 ID '{cam_id_str}'에 대한 캘리브레이션 데이터 업데이트 완료.")

            
            
    def update_from_dataset(self):
        cams = []
        calibration_paths = sorted(glob.glob(osp.join(self.example_path, 'calibration', '*.pkl')))
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
