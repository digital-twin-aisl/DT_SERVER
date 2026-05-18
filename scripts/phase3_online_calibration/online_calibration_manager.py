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
Online Calibration Manager — 온라인 Targetless Extrinsic Calibration 통합 관리자

Overlap Detector + Relative Pose Estimator + Factor Graph를 통합하여
센서 간 상대 변환을 주기적으로 추정하고 Pose Graph에 주입합니다.

작동 흐름:
  1. 주기적으로 (또는 이벤트 기반으로) 센서 쌍의 이미지 수집
  2. OverlapDetector로 overlapping 쌍 감지
  3. RelativePoseEstimator로 상대 T_i←j 추정
  4. PoseGraphManager에 inter-sensor factor로 주입
  5. 결과를 ROS 2 /sensor/extrinsic 토픽으로 발행

Temporal smoothing:
  - 매 추정 결과를 히스토리에 저장
  - 이상치(outlier) 필터링 (median-based)
  - 최근 N개 추정의 가중 평균으로 최종 extrinsic 산출

사용법:
    manager = OnlineCalibrationManager(
        pose_graph=pg_manager,
        sfm_map=sfm_map,
        device='cuda',
    )
    # 매 프레임 또는 주기적 호출
    results = manager.calibrate(
        images={'sensor_01': img1, 'sensor_02': img2},
        camera_intrinsics={'sensor_01': K1, 'sensor_02': K2},
    )
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import logging
import time
import threading

from scripts.phase3_online_calibration.overlap_detector import (
    OverlapDetector, OverlapResult,
)
from scripts.phase3_online_calibration.relative_pose_estimator import (
    RelativePoseEstimator, RelativePoseResult,
)

logger = logging.getLogger(__name__)

# Optional imports
try:
    from scripts.phase2_pose_graph.pose_graph_manager import PoseGraphManager
    POSE_GRAPH_AVAILABLE = True
except ImportError:
    POSE_GRAPH_AVAILABLE = False

try:
    from scripts.phase1_sfm.sfm_map import SfMMap
    SFM_MAP_AVAILABLE = True
except ImportError:
    SFM_MAP_AVAILABLE = False


@dataclass
class SensorPairHistory:
    """센서 쌍의 calibration 히스토리"""
    sensor_i: str
    sensor_j: str
    # 최근 추정 결과 (temporal window)
    estimates: deque = field(default_factory=lambda: deque(maxlen=50))
    # 현재 smoothed extrinsic
    T_i_from_j: Optional[np.ndarray] = None
    # 마지막 업데이트 시각
    last_update_time: float = 0.0
    # 연속 실패 횟수
    consecutive_failures: int = 0
    # 최초 감지 이후 총 추정 횟수
    total_estimates: int = 0
    # Overlap 상태
    is_overlapping: bool = False


@dataclass
class CalibrationResult:
    """단일 캘리브레이션 사이클의 결과"""
    # 감지된 overlapping 쌍
    overlapping_pairs: List[OverlapResult]
    # 성공적으로 추정된 상대 pose
    pose_estimates: List[RelativePoseResult]
    # Factor Graph에 주입된 수
    factors_injected: int
    # 처리 시간
    elapsed_seconds: float


class OnlineCalibrationManager:
    """
    온라인 Targetless Extrinsic Calibration 통합 관리자

    센서 쌍의 overlapping FOV를 감지하고, 상대 pose를 추정하며,
    결과를 Factor Graph에 주입하는 전체 파이프라인을 관리합니다.
    """

    def __init__(
        self,
        pose_graph: Optional['PoseGraphManager'] = None,
        sfm_map: Optional['SfMMap'] = None,
        device: str = 'cuda',
        max_keypoints: int = 2048,
        # Overlap detection thresholds
        min_matches: int = 30,
        min_match_ratio: float = 0.05,
        min_inlier_ratio: float = 0.5,
        # Relative pose estimation
        ransac_threshold: float = 1.0,
        min_pose_inliers: int = 20,
        # Temporal smoothing
        history_size: int = 50,
        outlier_threshold: float = 0.1,  # rotation diff threshold (rad)
        # Update control
        min_update_interval: float = 1.0,  # 최소 업데이트 간격 (초)
        max_consecutive_failures: int = 10,
    ):
        self._lock = threading.Lock()

        # ── Sub-modules ──────────────────────────────────────
        self.overlap_detector = OverlapDetector(
            device=device,
            max_keypoints=max_keypoints,
            min_matches=min_matches,
            min_match_ratio=min_match_ratio,
            min_inlier_ratio=min_inlier_ratio,
        )

        self.pose_estimator = RelativePoseEstimator(
            device=device,
            max_keypoints=max_keypoints,
            ransac_threshold=ransac_threshold,
            min_inliers=min_pose_inliers,
        )

        self.pose_graph = pose_graph
        self.sfm_map = sfm_map
        self._sfm_points_3d = None
        if sfm_map is not None and SFM_MAP_AVAILABLE:
            try:
                self._sfm_points_3d = sfm_map.get_all_3d_points()
                logger.info(f"Loaded {len(self._sfm_points_3d)} SfM 3D points for scale resolution")
            except Exception as e:
                logger.warning(f"Failed to load SfM 3D points: {e}")

        # ── Settings ─────────────────────────────────────────
        self.history_size = history_size
        self.outlier_threshold = outlier_threshold
        self.min_update_interval = min_update_interval
        self.max_consecutive_failures = max_consecutive_failures

        # ── State ────────────────────────────────────────────
        self._pair_histories: Dict[str, SensorPairHistory] = {}
        # 캐싱된 feature (센서별)
        self._feature_cache: Dict[str, dict] = {}

        # 통계
        self._total_cycles = 0
        self._total_factors = 0

        logger.info("OnlineCalibrationManager initialized")

    def _pair_key(self, sensor_i: str, sensor_j: str) -> str:
        """센서 쌍의 정렬된 키 (순서 무관)"""
        return f"{min(sensor_i, sensor_j)}__{max(sensor_i, sensor_j)}"

    def _get_history(self, sensor_i: str, sensor_j: str) -> SensorPairHistory:
        """센서 쌍의 히스토리 조회/생성"""
        key = self._pair_key(sensor_i, sensor_j)
        if key not in self._pair_histories:
            self._pair_histories[key] = SensorPairHistory(
                sensor_i=min(sensor_i, sensor_j),
                sensor_j=max(sensor_i, sensor_j),
                estimates=deque(maxlen=self.history_size),
            )
        return self._pair_histories[key]

    def calibrate(
        self,
        images: Dict[str, np.ndarray],
        camera_intrinsics: Dict[str, np.ndarray],
        sensor_timesteps: Optional[Dict[str, int]] = None,
    ) -> CalibrationResult:
        """
        한 사이클의 온라인 캘리브레이션 수행

        Args:
            images: {sensor_id: image_bgr} — 동시간 촬영 이미지
            camera_intrinsics: {sensor_id: K_3x3} — 카메라 intrinsic
            sensor_timesteps: {sensor_id: timestep} — Factor Graph 시점 (없으면 0)

        Returns:
            CalibrationResult
        """
        start_time = time.time()
        now = time.time()

        if sensor_timesteps is None:
            sensor_timesteps = {sid: 0 for sid in images}

        overlapping_pairs = []
        pose_estimates = []
        factors_injected = 0

        with self._lock:
            # ── Step 1: Feature 추출 (캐싱) ───────────────────
            for sensor_id, image in images.items():
                feats = self.overlap_detector.extract_features(image)
                self._feature_cache[sensor_id] = feats

            # ── Step 2: Overlap 감지 ──────────────────────────
            sensor_ids = list(images.keys())
            import itertools
            for id_i, id_j in itertools.combinations(sensor_ids, 2):
                history = self._get_history(id_i, id_j)

                # 최소 업데이트 간격 체크
                if now - history.last_update_time < self.min_update_interval:
                    continue

                # 연속 실패가 너무 많으면 건너뜀 (백오프)
                if history.consecutive_failures >= self.max_consecutive_failures:
                    # 10배 간격으로 재시도
                    backoff = self.min_update_interval * 10
                    if now - history.last_update_time < backoff:
                        continue

                K_i = camera_intrinsics.get(id_i)
                K_j = camera_intrinsics.get(id_j)

                if K_i is None or K_j is None:
                    continue

                # Feature 기반 overlap 분석
                feats_i = self._feature_cache.get(id_i)
                feats_j = self._feature_cache.get(id_j)

                if feats_i is None or feats_j is None:
                    continue

                overlap = self.overlap_detector.analyze_pair_from_features(
                    feats_i, feats_j, id_i, id_j, K_i,
                )

                if not overlap.is_overlapping:
                    history.is_overlapping = False
                    history.consecutive_failures += 1
                    history.last_update_time = now
                    continue

                overlapping_pairs.append(overlap)
                history.is_overlapping = True

                # ── Step 3: 상대 Pose 추정 ──────────────────
                pose_result = self.pose_estimator.estimate_from_matches(
                    matched_kps_i=overlap.matched_kps_i[overlap.inlier_mask],
                    matched_kps_j=overlap.matched_kps_j[overlap.inlier_mask],
                    camera_K_i=K_i,
                    camera_K_j=K_j,
                    sensor_id_i=id_i,
                    sensor_id_j=id_j,
                    known_scale=None,
                )

                if pose_result is None:
                    history.consecutive_failures += 1
                    history.last_update_time = now
                    continue

                # ── Step 4: Outlier 필터링 및 Temporal Smoothing ──
                if self._is_outlier(history, pose_result):
                    logger.info(
                        f"Outlier rejected: {id_i}↔{id_j} "
                        f"(rotation diff > {self.outlier_threshold:.3f} rad)"
                    )
                    history.last_update_time = now
                    continue

                # 히스토리에 추가
                history.estimates.append({
                    'T_i_from_j': pose_result.T_i_from_j.copy(),
                    'sigma_rot': pose_result.sigma_rotation,
                    'sigma_trans': pose_result.sigma_translation,
                    'inlier_ratio': pose_result.inlier_ratio,
                    'timestamp': now,
                })
                history.total_estimates += 1
                history.consecutive_failures = 0
                history.last_update_time = now

                # Smoothed extrinsic 업데이트
                smoothed_T = self._compute_smoothed_transform(history)
                history.T_i_from_j = smoothed_T

                pose_estimates.append(pose_result)

                # ── Step 5: Factor Graph에 주입 ──────────────
                if self.pose_graph is not None and POSE_GRAPH_AVAILABLE:
                    ti = sensor_timesteps.get(id_i, 0)
                    tj = sensor_timesteps.get(id_j, 0)

                    self.pose_graph.add_inter_sensor_factor(
                        sensor_id_i=id_i,
                        timestep_i=ti,
                        sensor_id_j=id_j,
                        timestep_j=tj,
                        relative_pose_4x4=smoothed_T,
                        sigma_rot=pose_result.sigma_rotation,
                        sigma_trans=pose_result.sigma_translation,
                    )
                    factors_injected += 1
                    self._total_factors += 1

                    logger.info(
                        f"Injected inter-sensor factor: {id_i}(t{ti})↔{id_j}(t{tj}), "
                        f"σ_rot={pose_result.sigma_rotation:.4f}, "
                        f"σ_trans={pose_result.sigma_translation:.4f}"
                    )

        self._total_cycles += 1
        elapsed = time.time() - start_time

        result = CalibrationResult(
            overlapping_pairs=overlapping_pairs,
            pose_estimates=pose_estimates,
            factors_injected=factors_injected,
            elapsed_seconds=elapsed,
        )

        logger.info(
            f"Calibration cycle #{self._total_cycles}: "
            f"{len(overlapping_pairs)} overlapping pairs, "
            f"{len(pose_estimates)} poses estimated, "
            f"{factors_injected} factors injected, "
            f"elapsed={elapsed:.3f}s"
        )

        return result

    def _is_outlier(self, history: SensorPairHistory, result: RelativePoseResult) -> bool:
        """
        새 추정이 outlier인지 판단

        이전 히스토리의 median과 비교하여 rotation 차이가
        threshold를 초과하면 outlier로 판단합니다.
        """
        if len(history.estimates) < 3:
            return False  # 충분한 히스토리가 없으면 통과

        # 이전 추정들의 rotation 추출
        from scipy.spatial.transform import Rotation
        prev_rotations = []
        for est in history.estimates:
            r = Rotation.from_matrix(est['T_i_from_j'][:3, :3])
            prev_rotations.append(r.as_rotvec())

        # 현재 추정의 rotation
        curr_rotvec = Rotation.from_matrix(result.T_i_from_j[:3, :3]).as_rotvec()

        # Median rotation vector
        prev_stack = np.array(prev_rotations)
        median_rotvec = np.median(prev_stack, axis=0)

        # Angular distance
        diff = np.linalg.norm(curr_rotvec - median_rotvec)
        return diff > self.outlier_threshold

    def _compute_smoothed_transform(self, history: SensorPairHistory) -> np.ndarray:
        """
        히스토리의 가중 평균으로 smoothed transform 산출

        최근 추정에 더 높은 가중치를 부여합니다 (exponential decay).
        """
        if len(history.estimates) == 0:
            return np.eye(4, dtype=np.float64)

        if len(history.estimates) == 1:
            return history.estimates[0]['T_i_from_j'].copy()

        from scipy.spatial.transform import Rotation, Slerp
        from scipy.spatial.transform import Rotation as R

        # 가중치: 최근 추정일수록 높은 가중치 + 높은 inlier_ratio
        timestamps = [e['timestamp'] for e in history.estimates]
        t_max = max(timestamps)
        t_min = min(timestamps)
        t_range = max(t_max - t_min, 1e-6)

        weights = []
        for e in history.estimates:
            # Time decay weight (0~1)
            time_w = np.exp(-2.0 * (t_max - e['timestamp']) / t_range)
            # Quality weight
            quality_w = e['inlier_ratio']
            weights.append(time_w * quality_w)

        weights = np.array(weights)
        weights = weights / weights.sum()

        # Translation: 가중 평균
        translations = np.array([e['T_i_from_j'][:3, 3] for e in history.estimates])
        avg_t = (weights[:, None] * translations).sum(axis=0)

        # Rotation: 가중 평균 (quaternion)
        rotations = [R.from_matrix(e['T_i_from_j'][:3, :3]) for e in history.estimates]
        quats = np.array([r.as_quat() for r in rotations])  # Nx4 (x,y,z,w)

        # Ensure all quaternions are in same hemisphere
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]

        avg_quat = (weights[:, None] * quats).sum(axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)

        avg_R = R.from_quat(avg_quat).as_matrix()

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = avg_R
        T[:3, 3] = avg_t

        return T

    # ─────────────────────────────────────────────────────────
    # 조회 API
    # ─────────────────────────────────────────────────────────

    def get_extrinsic(
        self, sensor_i: str, sensor_j: str
    ) -> Optional[np.ndarray]:
        """
        두 센서 간의 현재 최적 extrinsic 조회

        Returns:
            T_i←j (4x4) or None
        """
        key = self._pair_key(sensor_i, sensor_j)
        history = self._pair_histories.get(key)
        if history is None or history.T_i_from_j is None:
            return None

        # 순서 보정: 요청 순서와 저장 순서가 다를 수 있음
        if min(sensor_i, sensor_j) == sensor_i:
            return history.T_i_from_j.copy()
        else:
            return np.linalg.inv(history.T_i_from_j)

    def get_overlapping_pairs(self) -> List[Tuple[str, str]]:
        """현재 overlapping으로 감지된 센서 쌍 목록"""
        pairs = []
        for key, history in self._pair_histories.items():
            if history.is_overlapping:
                pairs.append((history.sensor_i, history.sensor_j))
        return pairs

    def get_stats(self) -> dict:
        """통계 반환"""
        return {
            'total_cycles': self._total_cycles,
            'total_factors_injected': self._total_factors,
            'num_tracked_pairs': len(self._pair_histories),
            'overlapping_pairs': len(self.get_overlapping_pairs()),
            'pair_details': {
                key: {
                    'sensor_i': h.sensor_i,
                    'sensor_j': h.sensor_j,
                    'is_overlapping': h.is_overlapping,
                    'total_estimates': h.total_estimates,
                    'history_size': len(h.estimates),
                    'consecutive_failures': h.consecutive_failures,
                }
                for key, h in self._pair_histories.items()
            },
        }

    def get_all_extrinsics(self) -> Dict[str, np.ndarray]:
        """
        모든 캘리브레이션된 센서 쌍의 extrinsic 조회

        Returns:
            {'sensor_i__sensor_j': T_i_from_j, ...}
        """
        extrinsics = {}
        for key, history in self._pair_histories.items():
            if history.T_i_from_j is not None:
                extrinsics[key] = history.T_i_from_j.copy()
        return extrinsics

    def to_transform_stamped_list(self) -> list:
        """
        모든 extrinsic을 ROS 2 TransformStamped 형식의 dict 리스트로 변환

        ROS 2 노드에서 /sensor/extrinsic 토픽으로 발행할 수 있는 형태입니다.
        """
        from scipy.spatial.transform import Rotation
        results = []
        for key, history in self._pair_histories.items():
            if history.T_i_from_j is None:
                continue
            T = history.T_i_from_j
            quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # x,y,z,w
            results.append({
                'parent_frame': history.sensor_i,
                'child_frame': history.sensor_j,
                'translation': T[:3, 3].tolist(),
                'rotation_xyzw': quat.tolist(),
            })
        return results
