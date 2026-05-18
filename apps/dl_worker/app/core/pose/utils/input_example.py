# This file is part of DT_SERVER.
# 
# DT_SERVER is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
# 
# DT_SERVER is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DT_SERVER; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

import os.path as osp
import glob
import cv2
import pickle
import numpy as np
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
from apps.dl_worker.app.core.pose.utils.transforms import get_affine_transform, get_scale

class ExampleDataset(IterableDataset):
    """
    Example Dataset class for FOCUS Inference        
    """
    def __init__(self, cfg, example_path=None, start_idx=0, end_idx=None):
        self.example_path = example_path
        self.video_paths = sorted(glob.glob(osp.join(self.example_path, 'hdVideos', '*.mp4')))
        self.calibration_paths = sorted(glob.glob(osp.join(self.example_path, 'calibration', '*.pkl')))

        self.start_idx = start_idx
        self.end_idx = end_idx

        self.camera = self.get_cam()
        self.caps = self.get_caps()

        self.transform = self.get_transform()
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        
        if  self.end_idx is None:
            self.end_idx = int(self.caps[0].get(cv2.CAP_PROP_FRAME_COUNT))        

    def get_transform(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        return transform

    def get_caps(self):
        caps = []
        for path in self.video_paths:
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_idx)
            caps.append(cap)
            
        return caps
        
    def get_cam(self):
        camera = []
        for path in self.calibration_paths:
            with open(path, "rb") as f:
                calib = pickle.load(f)
                
            M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            R, _ = cv2.Rodrigues(calib['rvec'])
            # R = R.dot(M)
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
            camera.append(cam)
        return camera
    
    def __iter__(self):
        idx = 0     
        while True:
            if idx >= self.end_idx - self.start_idx:
                break
            idx += 1
            inputs = []
            raw_images = []
            for cap in self.caps:
                ret, data_numpy = cap.read()
                frame_stamp = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if not ret:
                    print(f"Failed to read image")
                    assert False
                
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)     

                height, width, _ = data_numpy.shape
                c = np.array([width / 2.0, height / 2.0])
                s = get_scale((width, height), self.image_size)
                r = 0

                trans = get_affine_transform(c, s, r, self.image_size)
                input = cv2.warpAffine(
                    data_numpy,
                    trans, (int(self.image_size[0]), int(self.image_size[1])),
                    flags=cv2.INTER_LINEAR)
                input = self.transform(input)

                inputs.append(input)
                raw_images.append(data_numpy)

            meta = []

            for cam in self.camera:
                # m 딕셔너리 생성
                m = {
                'center': c,
                'scale': s,
                'rotation': r,
                'camera': cam
                }
                meta.append(m)
                
            yield raw_images, inputs, meta, frame_stamp
        for cap in self.caps:
            cap.release()