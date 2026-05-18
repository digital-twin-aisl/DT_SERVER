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

'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter


def get_index(indices, shape):
    batch_size = indices.shape[0]
    num_people = indices.shape[1]
    indices_x = (indices // (shape[1] * shape[2])).reshape(batch_size, num_people, -1)
    indices_y = ((indices % (shape[1] * shape[2])) // shape[2]).reshape(batch_size, num_people, -1)
    indices_z = (indices % shape[2]).reshape(batch_size, num_people, -1)
    indices = torch.cat([indices_x, indices_y, indices_z], dim=2)
    return indices


def max_pool(inputs, kernel=3):
    padding = (kernel - 1) // 2
    max = F.max_pool3d(inputs, kernel_size=kernel, stride=1, padding=padding)
    keep = (inputs == max).float()
    return keep * inputs

def refine_coordinates(cubes, peak_indices, kernel_size=3):
    """
    피크 지점 주변의 가중 평균을 계산하여 좌표를 보정합니다.
    - cubes: (B, D, H, W) 원본 신뢰도 맵
    - peak_indices: (B, N, 3) NMS를 통과한 정수 좌표들
    - kernel_size: 가중 평균을 계산할 주변 영역 크기
    """
    batch_size, num_peaks, _ = peak_indices.shape
    device = cubes.device
    pad = (kernel_size - 1) // 2
    
    # 패딩을 추가하여 경계 처리
    padded_cubes = F.pad(cubes, (pad, pad, pad, pad, pad, pad))
    
    refined_indices = []
    
    # 각 배치별로 처리
    for b in range(batch_size):
        batch_peaks = peak_indices[b] # (N, 3)
        batch_cube = padded_cubes[b] # (D+2*pad, H+2*pad, W+2*pad)
        
        # 로컬 윈도우 좌표 생성
        offset = torch.arange(-pad, pad + 1, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(offset, offset, offset, indexing='ij')
        
        batch_refined = []
        for i in range(num_peaks):
            # 피크의 정수 좌표 (z, y, x 순서일 수 있으니 확인 필요)
            px, py, pz = batch_peaks[i, 0], batch_peaks[i, 1], batch_peaks[i, 2]
            
            # 패딩된 큐브에서의 좌표로 변환
            px_pad, py_pad, pz_pad = px + pad, py + pad, pz + pad
            
            # 로컬 윈도우 추출
            local_window = batch_cube[px_pad-pad:px_pad+pad+1, py_pad-pad:py_pad+pad+1, pz_pad-pad:pz_pad+pad+1]
            
            # 정규화된 가중치 (총합이 1이 되도록)
            window_sum = torch.sum(local_window)
            if window_sum == 0: # 주변 값이 모두 0인 경우
                batch_refined.append(batch_peaks[i].float())
                continue
                
            weights = local_window / window_sum
            
            # 가중 평균 계산
            dx = torch.sum(weights * grid_x)
            dy = torch.sum(weights * grid_y)
            dz = torch.sum(weights * grid_z)
            
            # 원래 좌표에 보정값 더하기
            refined_x = px.float() + dx
            refined_y = py.float() + dy
            refined_z = pz.float() + dz
            
            batch_refined.append(torch.stack([refined_x, refined_y, refined_z]))
            
        refined_indices.append(torch.stack(batch_refined))
        
    return torch.stack(refined_indices)


def nms(root_cubes, max_num):
    batch_size = root_cubes.shape[0]
    # root_cubes_nms = torch.zeros_like(root_cubes, device=root_cubes.device)
    #
    # for b in range(batch_size):
    #     mx = torch.as_tensor(maximum_filter(root_cubes[b].detach().cpu().numpy(), size=3),
    #                          dtype=torch.float, device=root_cubes.device)
    #     root_cubes_nms[b] = (mx == root_cubes[b]).float() * root_cubes[b]
    root_cubes_nms = max_pool(root_cubes, kernel=5)
    root_cubes_nms_reshape = root_cubes_nms.reshape(batch_size, -1)
    topk_values, topk_index = root_cubes_nms_reshape.topk(max_num)
    topk_unravel_index = get_index(topk_index, root_cubes[0].shape)

    topk_unravel_index = refine_coordinates(root_cubes, topk_unravel_index)


    return topk_values, topk_unravel_index