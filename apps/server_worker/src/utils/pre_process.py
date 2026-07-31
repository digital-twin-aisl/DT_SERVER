import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time

from apps.server_worker.src.utils.transforms import get_affine_transform, get_scale

class PREPROCESS:
    def __init__(self, cfg):
        self.orig_image_size= np.array(cfg.NETWORK.IMAGE_SIZE_ORIG)
        self.image_size=np.array(cfg.NETWORK.IMAGE_SIZE)
        self.c = np.array([self.orig_image_size[0] / 2.0, self.orig_image_size[1] / 2.0])
        self.s = get_scale(self.orig_image_size, self.image_size)
        self.r = 0

        self._transform = self.get_transform()
        self.aff_transform = get_affine_transform(self.c, self.s, self.r, self.image_size)

        cfg.TRANSFORM = self.aff_transform
    
    def get_transform(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        return transform
    
    def __call__(self, inputs):
        t0 = time.time()
        if not isinstance(inputs, list):
            inputs = [[inputs]]
        elif inputs and not isinstance(inputs[0], list):
            inputs = [inputs]

        batch_size = len(inputs)
        if batch_size == 0:
            return torch.empty(0)
        num_cameras = len(inputs[0])
        # 바로 텐서로 변환하며 스택
        tensors = []
        for frame in inputs:
            for img in frame:
                img_aff = cv2.warpAffine(
                    img,
                    self.aff_transform,
                    (self.image_size[0], self.image_size[1]),
                    flags=cv2.INTER_LINEAR
                )
                tensors.append(self._transform(img_aff))
        final_tensor = torch.stack(tensors).to("cuda:0")

        print(f"preprocess time: {(time.time() - t0)*1000:.2f}")
        return final_tensor
    
    @property
    def transform(self):
        return self._transform