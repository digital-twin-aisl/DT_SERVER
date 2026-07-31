import numpy as np
import cv2
import torch

from apps.edge_client.src.utils.transforms import get_affine_transform, get_scale

class PREPROCESS:
    def __init__(self, cfg, device="cuda:0"):
        self.device = torch.device(device)

        self.orig_image_size = np.array(cfg.NETWORK.IMAGE_SIZE_ORIG)
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.c = np.array([self.orig_image_size[0] / 2.0, self.orig_image_size[1] / 2.0])
        self.s = get_scale(self.orig_image_size, self.image_size)
        self.r = 0

        self.aff_transform = get_affine_transform(self.c, self.s, self.r, self.image_size)
        cfg.TRANSFORM = self.aff_transform  # 기존 유지

        # mean/std는 GPU에 올려서 한 번에 normalize
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device)
        self.mean = mean.view(1, 3, 1, 1)
        self.std  = std.view(1, 3, 1, 1)

        # 재사용 가능한 pinned CPU 버퍼 (torch 텐서 + 그에 대응하는 numpy view)
        self._cpu_batch = None          # torch.Tensor, shape: (N, H, W, 3), uint8, pin_memory=True
        self._cpu_batch_np = None       # numpy view of _cpu_batch

    def __call__(self, inputs):
        # inputs: [img0, img1, ...] 또는 [[...], [...]] 그대로 지원
        if not isinstance(inputs, list):
            inputs = [[inputs]]
        elif inputs and not isinstance(inputs[0], list):
            inputs = [inputs]

        batch_size = len(inputs)
        if batch_size == 0:
            return []

        num_cameras = len(inputs[0])
        num_views = batch_size * num_cameras

        out_w, out_h = int(self.image_size[0]), int(self.image_size[1])
        H, W = out_h, out_w

        # 1) 전체를 한 번에 담을 pinned CPU 버퍼 (N, H, W, 3, uint8)
        needed_shape = (num_views, H, W, 3)
        if self._cpu_batch is None or tuple(self._cpu_batch.shape) != needed_shape:
            self._cpu_batch = torch.empty(
                needed_shape,
                dtype=torch.uint8,
                pin_memory=True,   # pinned memory
            )
            # torch CPU 텐서의 numpy view (zero-copy)
            self._cpu_batch_np = self._cpu_batch.numpy()

        np_batch = self._cpu_batch_np

        # 2) 각 프레임을 warpAffine해서 np_batch[idx]에 직접 써 넣기
        idx = 0
        for frame in inputs:           # frame: [cam0_img, cam1_img, ...]
            for img in frame:          # img: (H_orig, W_orig, 3), uint8
                dst = np_batch[idx]    # (H, W, 3) view
                cv2.warpAffine(
                    img,
                    self.aff_transform,
                    (out_w, out_h),
                    dst=dst,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                idx += 1

        # 3) pinned CPU 텐서를 한 번에 GPU로 전송 + float32 + /255
        batch = self._cpu_batch.to(self.device, dtype=torch.float32, non_blocking=True)
        # (N, H, W, 3) -> (N, 3, H, W)
        batch = batch.permute(0, 3, 1, 2)
        batch.mul_(1.0 / 255.0)

        # 4) normalize를 GPU에서 배치 한 번에
        batch.sub_(self.mean)
        batch.div_(self.std)

        # 5) 모델 인터페이스: 리스트 of (1, 3, H, W)
        views = [batch[i:i+1] for i in range(num_views)]
        return views

    @property
    def transform(self):
        return None

    @property
    def aff_transform_matrix(self):
        return self.aff_transform