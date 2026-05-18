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
Overlap Detector — 센서 쌍의 Overlapping FOV 자동 감지

두 센서의 동시간 프레임에서 SuperPoint 특징점을 추출하고
LightGlue로 매칭하여, 충분한 매치가 발생하는 센서 쌍을
자동으로 감지합니다.

감지 기준:
  - 공유 feature 매치 수 ≥ min_matches (default: 30)
  - 매치 비율 (matches / min(kps_i, kps_j)) ≥ min_match_ratio (default: 0.05)
  - Essential matrix RANSAC inlier 비율 ≥ min_inlier_ratio (default: 0.5)

사용법:
    detector = OverlapDetector(device='cuda')
    pairs = detector.detect_overlapping_pairs(
        images={'sensor_01': img1, 'sensor_02': img2, ...}
    )
    # pairs = [('sensor_01', 'sensor_02'), ...]
"""
import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import itertools

logger = logging.getLogger(__name__)

try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import numpy_image_to_torch
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    logger.warning(
        "LightGlue not available. Install with: pip install lightglue. "
        "OverlapDetector will not work."
    )


@dataclass
class OverlapResult:
    """센서 쌍의 overlap 분석 결과"""
    sensor_i: str
    sensor_j: str
    num_matches: int
    match_ratio: float           # matches / min(kps_i, kps_j)
    num_inliers: int             # Essential matrix RANSAC inliers
    inlier_ratio: float          # inliers / matches
    is_overlapping: bool
    # 매치된 키포인트 좌표 (상대 pose 추정에 활용)
    matched_kps_i: Optional[np.ndarray] = None  # Nx2
    matched_kps_j: Optional[np.ndarray] = None  # Nx2
    inlier_mask: Optional[np.ndarray] = None    # N bool


class OverlapDetector:
    """
    SuperPoint + LightGlue 기반 Overlapping FOV 자동 감지기

    두 센서 이미지 간의 feature matching 통계를 기반으로
    FOV가 겹치는 센서 쌍을 자동으로 식별합니다.
    """

    def __init__(
        self,
        device: str = 'cuda',
        max_keypoints: int = 2048,
        min_matches: int = 30,
        min_match_ratio: float = 0.05,
        min_inlier_ratio: float = 0.5,
    ):
        """
        Args:
            device: 'cuda' or 'cpu'
            max_keypoints: SuperPoint 최대 키포인트 수
            min_matches: 최소 매치 수 (이하면 non-overlapping)
            min_match_ratio: 최소 매치 비율
            min_inlier_ratio: Essential matrix RANSAC 최소 inlier 비율
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.min_matches = min_matches
        self.min_match_ratio = min_match_ratio
        self.min_inlier_ratio = min_inlier_ratio

        if not LIGHTGLUE_AVAILABLE:
            logger.error("LightGlue not available. OverlapDetector disabled.")
            self.extractor = None
            self.matcher = None
            return

        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        logger.info(
            f"OverlapDetector initialized: device={self.device}, "
            f"max_kp={max_keypoints}, min_matches={min_matches}"
        )

    @torch.no_grad()
    def extract_features(self, image: np.ndarray) -> dict:
        """
        이미지에서 SuperPoint feature 추출

        Args:
            image: BGR or RGB numpy array (HxWx3)

        Returns:
            SuperPoint feature dict (keypoints, descriptors, scores)
        """
        if self.extractor is None:
            return {}

        # BGR → grayscale → torch tensor [1, 1, H, W]
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Normalize to [0, 1] float32
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, H, W]

        feats = self.extractor({'image': tensor})
        return feats

    @torch.no_grad()
    def match_features(self, feats_i: dict, feats_j: dict) -> np.ndarray:
        """
        두 feature set 간 LightGlue 매칭

        Returns:
            Nx2 array of (idx_i, idx_j) valid match pairs
        """
        if self.matcher is None:
            return np.array([], dtype=np.int64).reshape(0, 2)

        result = self.matcher({'image0': feats_i, 'image1': feats_j})
        matches = result['matches'][0].cpu().numpy()
        # LightGlue returns Mx2, filter valid
        valid = matches[:, 0] >= 0
        return matches[valid]

    def analyze_pair(
        self,
        image_i: np.ndarray,
        image_j: np.ndarray,
        sensor_id_i: str,
        sensor_id_j: str,
        camera_K: Optional[np.ndarray] = None,
    ) -> OverlapResult:
        """
        두 센서 이미지 간 overlap 분석

        Args:
            image_i: 센서 i의 이미지 (BGR, HxWx3)
            image_j: 센서 j의 이미지 (BGR, HxWx3)
            sensor_id_i: 센서 i 식별자
            sensor_id_j: 센서 j 식별자
            camera_K: 3x3 intrinsic matrix (Essential matrix 추정용, None → 근사)

        Returns:
            OverlapResult with overlap analysis
        """
        if self.extractor is None:
            return OverlapResult(
                sensor_i=sensor_id_i, sensor_j=sensor_id_j,
                num_matches=0, match_ratio=0.0,
                num_inliers=0, inlier_ratio=0.0,
                is_overlapping=False,
            )

        # 1. Feature 추출
        feats_i = self.extract_features(image_i)
        feats_j = self.extract_features(image_j)

        kps_i = feats_i['keypoints'][0].cpu().numpy()  # Nx2
        kps_j = feats_j['keypoints'][0].cpu().numpy()

        # 2. Feature 매칭
        matches = self.match_features(feats_i, feats_j)
        num_matches = len(matches)

        min_kps = min(len(kps_i), len(kps_j))
        match_ratio = num_matches / max(min_kps, 1)

        if num_matches < self.min_matches:
            return OverlapResult(
                sensor_i=sensor_id_i, sensor_j=sensor_id_j,
                num_matches=num_matches, match_ratio=match_ratio,
                num_inliers=0, inlier_ratio=0.0,
                is_overlapping=False,
            )

        # 3. 매칭된 키포인트 좌표
        matched_kps_i = kps_i[matches[:, 0]]  # Nx2
        matched_kps_j = kps_j[matches[:, 1]]  # Nx2

        # 4. Essential/Fundamental matrix RANSAC으로 geometric verification
        if camera_K is not None:
            # Essential matrix (calibrated case)
            E, inlier_mask = cv2.findEssentialMat(
                matched_kps_i, matched_kps_j, camera_K,
                method=cv2.RANSAC, prob=0.999, threshold=1.0,
            )
        else:
            # Fundamental matrix (uncalibrated case)
            F, inlier_mask = cv2.findFundamentalMat(
                matched_kps_i, matched_kps_j,
                method=cv2.FM_RANSAC, ransacReprojThreshold=3.0,
                confidence=0.999,
            )

        if inlier_mask is None:
            return OverlapResult(
                sensor_i=sensor_id_i, sensor_j=sensor_id_j,
                num_matches=num_matches, match_ratio=match_ratio,
                num_inliers=0, inlier_ratio=0.0,
                is_overlapping=False,
                matched_kps_i=matched_kps_i,
                matched_kps_j=matched_kps_j,
            )

        inlier_mask = inlier_mask.flatten().astype(bool)
        num_inliers = int(inlier_mask.sum())
        inlier_ratio = num_inliers / max(num_matches, 1)

        is_overlapping = (
            num_matches >= self.min_matches
            and match_ratio >= self.min_match_ratio
            and inlier_ratio >= self.min_inlier_ratio
        )

        logger.info(
            f"Overlap analysis {sensor_id_i}↔{sensor_id_j}: "
            f"matches={num_matches}, ratio={match_ratio:.3f}, "
            f"inliers={num_inliers}, inlier_ratio={inlier_ratio:.3f} "
            f"→ {'OVERLAP' if is_overlapping else 'NO OVERLAP'}"
        )

        return OverlapResult(
            sensor_i=sensor_id_i,
            sensor_j=sensor_id_j,
            num_matches=num_matches,
            match_ratio=match_ratio,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            is_overlapping=is_overlapping,
            matched_kps_i=matched_kps_i,
            matched_kps_j=matched_kps_j,
            inlier_mask=inlier_mask,
        )

    def detect_overlapping_pairs(
        self,
        images: Dict[str, np.ndarray],
        camera_K: Optional[np.ndarray] = None,
    ) -> List[OverlapResult]:
        """
        N개 센서 이미지에서 모든 Overlapping 쌍을 감지

        Args:
            images: {sensor_id: image_bgr} dictionary
            camera_K: 공통 intrinsic (None이면 Fundamental matrix 사용)

        Returns:
            Overlapping 쌍의 OverlapResult 리스트
        """
        sensor_ids = list(images.keys())
        overlapping_pairs = []
        total_pairs = 0

        for id_i, id_j in itertools.combinations(sensor_ids, 2):
            total_pairs += 1
            result = self.analyze_pair(
                images[id_i], images[id_j], id_i, id_j, camera_K
            )
            if result.is_overlapping:
                overlapping_pairs.append(result)

        logger.info(
            f"Overlap detection complete: {len(overlapping_pairs)}/{total_pairs} "
            f"overlapping pairs found from {len(sensor_ids)} sensors"
        )

        return overlapping_pairs

    def analyze_pair_from_features(
        self,
        feats_i: dict,
        feats_j: dict,
        sensor_id_i: str,
        sensor_id_j: str,
        camera_K: Optional[np.ndarray] = None,
    ) -> OverlapResult:
        """
        이미 추출된 feature로부터 overlap 분석 (재추출 없이)

        feature를 캐싱하여 재사용할 때 유용합니다.
        """
        if self.matcher is None:
            return OverlapResult(
                sensor_i=sensor_id_i, sensor_j=sensor_id_j,
                num_matches=0, match_ratio=0.0,
                num_inliers=0, inlier_ratio=0.0,
                is_overlapping=False,
            )

        kps_i = feats_i['keypoints'][0].cpu().numpy()
        kps_j = feats_j['keypoints'][0].cpu().numpy()

        matches = self.match_features(feats_i, feats_j)
        num_matches = len(matches)
        min_kps = min(len(kps_i), len(kps_j))
        match_ratio = num_matches / max(min_kps, 1)

        if num_matches < self.min_matches:
            return OverlapResult(
                sensor_i=sensor_id_i, sensor_j=sensor_id_j,
                num_matches=num_matches, match_ratio=match_ratio,
                num_inliers=0, inlier_ratio=0.0,
                is_overlapping=False,
            )

        matched_kps_i = kps_i[matches[:, 0]]
        matched_kps_j = kps_j[matches[:, 1]]

        if camera_K is not None:
            _, inlier_mask = cv2.findEssentialMat(
                matched_kps_i, matched_kps_j, camera_K,
                method=cv2.RANSAC, prob=0.999, threshold=1.0,
            )
        else:
            _, inlier_mask = cv2.findFundamentalMat(
                matched_kps_i, matched_kps_j,
                method=cv2.FM_RANSAC, ransacReprojThreshold=3.0,
                confidence=0.999,
            )

        if inlier_mask is None:
            return OverlapResult(
                sensor_i=sensor_id_i, sensor_j=sensor_id_j,
                num_matches=num_matches, match_ratio=match_ratio,
                num_inliers=0, inlier_ratio=0.0,
                is_overlapping=False,
                matched_kps_i=matched_kps_i,
                matched_kps_j=matched_kps_j,
            )

        inlier_mask = inlier_mask.flatten().astype(bool)
        num_inliers = int(inlier_mask.sum())
        inlier_ratio = num_inliers / max(num_matches, 1)

        is_overlapping = (
            num_matches >= self.min_matches
            and match_ratio >= self.min_match_ratio
            and inlier_ratio >= self.min_inlier_ratio
        )

        return OverlapResult(
            sensor_i=sensor_id_i,
            sensor_j=sensor_id_j,
            num_matches=num_matches,
            match_ratio=match_ratio,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            is_overlapping=is_overlapping,
            matched_kps_i=matched_kps_i,
            matched_kps_j=matched_kps_j,
            inlier_mask=inlier_mask,
        )
