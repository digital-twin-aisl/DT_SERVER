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

"""
Relative Pose Estimator — SuperPoint+LightGlue 기반 센서 간 상대 Pose 추정

Overlapping FOV가 확인된 센서 쌍에 대해 정밀한 상대 변환 T_i←j를
추정합니다.

추정 파이프라인:
  1. SuperPoint + LightGlue로 feature 매칭
  2. Essential Matrix 추정 (RANSAC)
  3. Essential → R, t 분해 (cheirality check)
  4. Scale resolution:
     a) SfM 3D 맵 포인트가 있는 경우: triangulated depth 비교
     b) 없는 경우: 알려진 세계 구조물 크기 또는 1.0 (scale-free)

결과:
  - T_i←j: 4x4 transform (sensor_j를 sensor_i 좌표계로 변환)
  - 불확실성 추정 (covariance-like)
  - inlier 수 및 reproj error

사용법:
    estimator = RelativePoseEstimator(device='cuda')
    result = estimator.estimate(
        image_i, image_j, camera_K,
        sensor_id_i='cam_01', sensor_id_j='cam_02',
    )
    T_ij = result.T_i_from_j  # 4x4
"""
import cv2
import numpy as np
import torch
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    from lightglue import LightGlue, SuperPoint
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    logger.warning("LightGlue not available.")


@dataclass
class RelativePoseResult:
    """센서 간 상대 pose 추정 결과"""
    sensor_i: str
    sensor_j: str
    # T_i←j: sensor_j 좌표를 sensor_i 좌표로 변환하는 4x4 행렬
    T_i_from_j: np.ndarray              # 4x4
    # 역변환
    T_j_from_i: np.ndarray              # 4x4
    # 품질 지표
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    mean_reproj_error: float             # 재투영 오류 (px)
    # 불확실성 (6-DOF sigma 추정)
    sigma_rotation: float                # rotation 불확실성 (rad)
    sigma_translation: float             # translation 불확실성 (m)
    # Scale 정보
    scale_resolved: bool                 # scale이 해결되었는지
    scale_factor: float                  # 적용된 scale
    # 원본 매칭 데이터 (디버깅/시각화용)
    matched_kps_i: Optional[np.ndarray] = None  # Nx2
    matched_kps_j: Optional[np.ndarray] = None  # Nx2
    inlier_mask: Optional[np.ndarray] = None    # N bool


class RelativePoseEstimator:
    """
    SuperPoint + LightGlue 기반 센서 쌍 상대 Pose 추정기

    두 센서의 동시간 이미지에서 epipolar geometry를 이용하여
    상대 변환 T_i←j를 추정합니다.
    """

    def __init__(
        self,
        device: str = 'cuda',
        max_keypoints: int = 2048,
        ransac_threshold: float = 1.0,
        ransac_confidence: float = 0.9999,
        min_inliers: int = 20,
    ):
        """
        Args:
            device: 'cuda' or 'cpu'
            max_keypoints: SuperPoint 최대 키포인트 수
            ransac_threshold: Essential matrix RANSAC threshold (px)
            ransac_confidence: RANSAC confidence
            min_inliers: 최소 inlier 수
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.ransac_threshold = ransac_threshold
        self.ransac_confidence = ransac_confidence
        self.min_inliers = min_inliers

        if not LIGHTGLUE_AVAILABLE:
            self.extractor = None
            self.matcher = None
            return

        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        logger.info(
            f"RelativePoseEstimator initialized: device={self.device}, "
            f"max_kp={max_keypoints}, ransac_thresh={ransac_threshold}"
        )

    @torch.no_grad()
    def _extract_features(self, image: np.ndarray) -> dict:
        """이미지에서 SuperPoint feature 추출"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        return self.extractor({'image': tensor})

    @torch.no_grad()
    def _match_features(self, feats_i: dict, feats_j: dict) -> np.ndarray:
        """LightGlue 매칭 → Nx2 valid match pairs"""
        result = self.matcher({'image0': feats_i, 'image1': feats_j})
        matches = result['matches'][0].cpu().numpy()
        valid = matches[:, 0] >= 0
        return matches[valid]

    def estimate(
        self,
        image_i: np.ndarray,
        image_j: np.ndarray,
        camera_K_i: np.ndarray,
        camera_K_j: Optional[np.ndarray] = None,
        sensor_id_i: str = 'sensor_i',
        sensor_id_j: str = 'sensor_j',
        known_scale: Optional[float] = None,
        sfm_points_3d: Optional[np.ndarray] = None,
    ) -> Optional[RelativePoseResult]:
        """
        두 센서 이미지로부터 상대 pose T_i←j 추정

        Args:
            image_i: 센서 i 이미지 (BGR)
            image_j: 센서 j 이미지 (BGR)
            camera_K_i: 센서 i의 3x3 intrinsic matrix
            camera_K_j: 센서 j의 3x3 intrinsic (None → camera_K_i와 동일 가정)
            sensor_id_i: 센서 i 식별자
            sensor_id_j: 센서 j 식별자
            known_scale: 알려진 scale factor (None → scale-free 또는 SfM으로 결정)
            sfm_points_3d: SfM 3D 포인트 (scale resolution에 사용)

        Returns:
            RelativePoseResult or None (추정 실패 시)
        """
        if self.extractor is None:
            logger.error("LightGlue not available")
            return None

        if camera_K_j is None:
            camera_K_j = camera_K_i

        # 1. Feature 추출 및 매칭
        feats_i = self._extract_features(image_i)
        feats_j = self._extract_features(image_j)
        matches = self._match_features(feats_i, feats_j)

        if len(matches) < self.min_inliers:
            logger.warning(
                f"Not enough matches ({len(matches)}) for "
                f"{sensor_id_i}↔{sensor_id_j}"
            )
            return None

        kps_i = feats_i['keypoints'][0].cpu().numpy()
        kps_j = feats_j['keypoints'][0].cpu().numpy()
        matched_kps_i = kps_i[matches[:, 0]]
        matched_kps_j = kps_j[matches[:, 1]]

        # 2. Essential Matrix 추정 (RANSAC)
        E, inlier_mask = cv2.findEssentialMat(
            matched_kps_i, matched_kps_j, camera_K_i,
            method=cv2.RANSAC,
            prob=self.ransac_confidence,
            threshold=self.ransac_threshold,
        )

        if E is None or inlier_mask is None:
            logger.warning(f"Essential matrix estimation failed for {sensor_id_i}↔{sensor_id_j}")
            return None

        inlier_mask = inlier_mask.flatten().astype(bool)
        num_inliers = int(inlier_mask.sum())

        if num_inliers < self.min_inliers:
            logger.warning(
                f"Not enough inliers ({num_inliers}) for "
                f"{sensor_id_i}↔{sensor_id_j}"
            )
            return None

        # 3. Essential → R, t 분해 (cheirality check 포함)
        num_good, R, t, mask_pose = cv2.recoverPose(
            E, matched_kps_i[inlier_mask], matched_kps_j[inlier_mask],
            camera_K_i,
        )

        if num_good < self.min_inliers:
            logger.warning(
                f"recoverPose failed: only {num_good} points pass cheirality for "
                f"{sensor_id_i}↔{sensor_id_j}"
            )
            return None

        # t는 unit vector (scale = 1)
        t = t.flatten()

        # 4. Scale resolution
        scale_resolved = False
        scale_factor = 1.0

        if known_scale is not None:
            scale_factor = known_scale
            scale_resolved = True
        elif sfm_points_3d is not None and len(sfm_points_3d) > 0:
            # SfM 포인트를 이용한 scale 결정
            scale_factor = self._resolve_scale_from_sfm(
                matched_kps_i[inlier_mask], matched_kps_j[inlier_mask],
                camera_K_i, camera_K_j, R, t, sfm_points_3d,
            )
            scale_resolved = True if scale_factor > 0 else False

        t_scaled = t * scale_factor

        # 5. T_i←j 구성
        # R, t는 camera_j를 camera_i 좌표계로 변환하는 변환
        T_i_from_j = np.eye(4, dtype=np.float64)
        T_i_from_j[:3, :3] = R
        T_i_from_j[:3, 3] = t_scaled

        T_j_from_i = np.linalg.inv(T_i_from_j)

        # 6. 재투영 오류 계산
        mean_reproj = self._compute_reproj_error(
            matched_kps_i[inlier_mask], matched_kps_j[inlier_mask],
            camera_K_i, camera_K_j, R, t_scaled,
        )

        # 7. 불확실성 추정 (heuristic)
        inlier_ratio = num_inliers / len(matches)
        sigma_rot = max(0.005, 0.05 * (1.0 - inlier_ratio))    # 더 높은 inlier → 더 낮은 σ
        sigma_trans = max(0.01, 0.1 * (1.0 - inlier_ratio))
        if not scale_resolved:
            sigma_trans *= 5.0  # scale 미해결 시 translation 불확실성 증가

        logger.info(
            f"Relative pose {sensor_id_i}←{sensor_id_j}: "
            f"matches={len(matches)}, inliers={num_inliers} "
            f"({inlier_ratio:.1%}), reproj={mean_reproj:.2f}px, "
            f"scale={'resolved' if scale_resolved else 'unresolved'} "
            f"({scale_factor:.3f})"
        )

        return RelativePoseResult(
            sensor_i=sensor_id_i,
            sensor_j=sensor_id_j,
            T_i_from_j=T_i_from_j,
            T_j_from_i=T_j_from_i,
            num_matches=len(matches),
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            mean_reproj_error=mean_reproj,
            sigma_rotation=sigma_rot,
            sigma_translation=sigma_trans,
            scale_resolved=scale_resolved,
            scale_factor=scale_factor,
            matched_kps_i=matched_kps_i,
            matched_kps_j=matched_kps_j,
            inlier_mask=inlier_mask,
        )

    def estimate_from_matches(
        self,
        matched_kps_i: np.ndarray,
        matched_kps_j: np.ndarray,
        camera_K_i: np.ndarray,
        camera_K_j: Optional[np.ndarray] = None,
        sensor_id_i: str = 'sensor_i',
        sensor_id_j: str = 'sensor_j',
        known_scale: Optional[float] = None,
    ) -> Optional[RelativePoseResult]:
        """
        이미 매칭된 키포인트로부터 상대 pose 추정

        OverlapDetector의 결과를 직접 사용할 때 유용합니다.
        """
        if camera_K_j is None:
            camera_K_j = camera_K_i

        if len(matched_kps_i) < self.min_inliers:
            return None

        E, inlier_mask = cv2.findEssentialMat(
            matched_kps_i, matched_kps_j, camera_K_i,
            method=cv2.RANSAC,
            prob=self.ransac_confidence,
            threshold=self.ransac_threshold,
        )

        if E is None or inlier_mask is None:
            return None

        inlier_mask = inlier_mask.flatten().astype(bool)
        num_inliers = int(inlier_mask.sum())

        if num_inliers < self.min_inliers:
            return None

        num_good, R, t, _ = cv2.recoverPose(
            E, matched_kps_i[inlier_mask], matched_kps_j[inlier_mask],
            camera_K_i,
        )

        if num_good < self.min_inliers:
            return None

        t = t.flatten()
        scale_factor = known_scale if known_scale is not None else 1.0
        scale_resolved = known_scale is not None
        t_scaled = t * scale_factor

        T_i_from_j = np.eye(4, dtype=np.float64)
        T_i_from_j[:3, :3] = R
        T_i_from_j[:3, 3] = t_scaled

        T_j_from_i = np.linalg.inv(T_i_from_j)

        mean_reproj = self._compute_reproj_error(
            matched_kps_i[inlier_mask], matched_kps_j[inlier_mask],
            camera_K_i, camera_K_j, R, t_scaled,
        )

        inlier_ratio = num_inliers / len(matched_kps_i)
        sigma_rot = max(0.005, 0.05 * (1.0 - inlier_ratio))
        sigma_trans = max(0.01, 0.1 * (1.0 - inlier_ratio))
        if not scale_resolved:
            sigma_trans *= 5.0

        return RelativePoseResult(
            sensor_i=sensor_id_i,
            sensor_j=sensor_id_j,
            T_i_from_j=T_i_from_j,
            T_j_from_i=T_j_from_i,
            num_matches=len(matched_kps_i),
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            mean_reproj_error=mean_reproj,
            sigma_rotation=sigma_rot,
            sigma_translation=sigma_trans,
            scale_resolved=scale_resolved,
            scale_factor=scale_factor,
            matched_kps_i=matched_kps_i,
            matched_kps_j=matched_kps_j,
            inlier_mask=inlier_mask,
        )

    @staticmethod
    def _resolve_scale_from_sfm(
        kps_i: np.ndarray,
        kps_j: np.ndarray,
        K_i: np.ndarray,
        K_j: np.ndarray,
        R: np.ndarray,
        t_unit: np.ndarray,
        sfm_points_3d: np.ndarray,
    ) -> float:
        """
        SfM 3D 맵 포인트를 이용하여 translation의 절대 scale을 결정

        삼각측량으로 구한 상대 포인트 깊이와 SfM 맵 포인트의 깊이를 비교합니다.
        """
        # Projection matrices: camera_i = [I|0], camera_j = [R|t]
        P1 = K_i @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K_j @ np.hstack([R, t_unit.reshape(3, 1)])

        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, kps_i.T, kps_j.T)
        points_3d = (points_4d[:3] / points_4d[3:]).T  # Nx3

        # Filter valid points (positive depth in both cameras)
        depths_1 = points_3d[:, 2]
        depths_2 = (R @ points_3d.T + t_unit.reshape(3, 1))[2, :]
        valid = (depths_1 > 0) & (depths_2 > 0)

        if valid.sum() < 5:
            return 1.0

        # 삼각측량 포인트와 가장 가까운 SfM 포인트 간 거리 비교
        tri_points = points_3d[valid]
        # 중간값 깊이 비교 (robust)
        tri_median_depth = np.median(np.linalg.norm(tri_points, axis=1))

        # SfM 포인트의 median depth (원점 기준)
        sfm_median_depth = np.median(np.linalg.norm(sfm_points_3d, axis=1))

        if tri_median_depth < 1e-6:
            return 1.0

        scale = sfm_median_depth / tri_median_depth
        logger.info(
            f"Scale resolved from SfM: tri_depth={tri_median_depth:.3f}, "
            f"sfm_depth={sfm_median_depth:.3f} → scale={scale:.3f}"
        )
        return float(scale)

    @staticmethod
    def _compute_reproj_error(
        kps_i: np.ndarray,
        kps_j: np.ndarray,
        K_i: np.ndarray,
        K_j: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> float:
        """
        상대 pose의 재투영 오류 계산

        삼각측량 → 양쪽 카메라로 재투영 → 평균 오류
        """
        P1 = K_i @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K_j @ np.hstack([R, t.reshape(3, 1)])

        try:
            points_4d = cv2.triangulatePoints(P1, P2, kps_i.T, kps_j.T)
            points_3d = (points_4d[:3] / points_4d[3:]).T

            # Reproject to camera i
            proj_i = (K_i @ points_3d.T).T
            proj_i = proj_i[:, :2] / proj_i[:, 2:3]

            # Reproject to camera j
            pts_j = (R @ points_3d.T + t.reshape(3, 1)).T
            proj_j = (K_j @ pts_j.T).T
            proj_j = proj_j[:, :2] / proj_j[:, 2:3]

            err_i = np.linalg.norm(kps_i - proj_i, axis=1)
            err_j = np.linalg.norm(kps_j - proj_j, axis=1)

            return float(np.mean(np.concatenate([err_i, err_j])))
        except Exception:
            return float('inf')
