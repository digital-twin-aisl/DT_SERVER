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
Pose Graph Manager — GTSAM iSAM2 기반 연속 Pose 최적화 엔진

모든 센서(고정/이동)의 관측을 Factor Graph로 통합하여
MAP(Maximum A Posteriori) 추정을 수행합니다.

Factor 유형:
  - PriorFactor: 고정 센서 pose, relocalization 결과, GPS/UWB
  - BetweenFactor: odometry, inter-sensor constraint, loop closure
  
iSAM2를 사용하여 incremental update → O(k log N) 복잡도

사용법:
    manager = PoseGraphManager()
    manager.add_fixed_sensor_prior("sensor_fixed_01", pose, cov)
    manager.add_odometry("robot_01", t0, t1, delta_pose, cov)
    manager.add_relocalization_prior("robot_01", t, pose, cov)
    manager.optimize()
    result = manager.get_current_pose("robot_01", t)
"""
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from collections import deque
import logging
import threading
import time

logger = logging.getLogger(__name__)

try:
    import gtsam
    from gtsam import (
        NonlinearFactorGraph,
        Values,
        Pose3,
        Rot3,
        Point3,
        PriorFactorPose3,
        BetweenFactorPose3,
        noiseModel,
    )
    from gtsam import ISAM2, ISAM2Params
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    logger.warning(
        "GTSAM not available. Install with: pip install gtsam. "
        "Running in stub mode."
    )


def numpy_to_pose3(T: np.ndarray) -> 'Pose3':
    """4x4 numpy transform → gtsam.Pose3"""
    R = Rot3(T[:3, :3])
    t = Point3(T[0, 3], T[1, 3], T[2, 3])
    return Pose3(R, t)


def pose3_to_numpy(pose: 'Pose3') -> np.ndarray:
    """gtsam.Pose3 → 4x4 numpy transform"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = pose.translation()
    return T


def pose3_to_quat_pos(pose: 'Pose3') -> Tuple[np.ndarray, np.ndarray]:
    """gtsam.Pose3 → (quaternion_wxyz, position_xyz)"""
    q = pose.rotation().toQuaternion()
    pos = pose.translation()
    return (
        np.array([q.w(), q.x(), q.y(), q.z()]),
        np.array([pos[0], pos[1], pos[2]])
    )


def make_diagonal_noise(sigmas_6dof: np.ndarray) -> 'noiseModel.Diagonal':
    """
    6-DOF 대각 노이즈 모델 생성
    
    Args:
        sigmas_6dof: [rx, ry, rz, tx, ty, tz] 표준편차 (rad, m)
    """
    return noiseModel.Diagonal.Sigmas(sigmas_6dof)


@dataclass
class SensorState:
    """개별 센서의 상태 추적"""
    sensor_id: str
    is_fixed: bool
    # 고정 센서: 단일 symbol key
    # 이동 센서: timestep별 symbol key
    key_map: Dict[int, int] = field(default_factory=dict)  # timestep → gtsam key
    latest_timestep: int = 0
    # 고정 센서의 경우 key는 하나만
    fixed_key: Optional[int] = None


class PoseGraphManager:
    """
    GTSAM iSAM2 기반 Pose Graph 관리자
    
    고정 센서(CCTV)와 이동형 센서(로봇, 드론)를 동시에 관리하며,
    각종 factor를 추가하고 incremental 최적화를 수행합니다.
    """

    # Symbol character mapping for sensors
    # 'a'~'z' → 최대 26개 센서 (확장 가능)
    _SENSOR_CHARS = 'abcdefghijklmnopqrstuvwxyz'

    def __init__(
        self,
        relinearize_threshold: float = 0.1,
        relinearize_skip: int = 10,
        enable_partial_relinearization: bool = True,
        adaptive_relinearize: bool = False,
        target_update_ms: float = 50.0,
        timing_window: int = 50,
    ):
        """
        Args:
            relinearize_threshold: iSAM2 재선형화 임계값 (작을수록 정확, 느림)
            relinearize_skip: N번 업데이트마다 재선형화 체크
            enable_partial_relinearization: 부분 재선형화 허용
            adaptive_relinearize: True면 사이클 타이밍 기반으로 relinearize_skip 자동 조정
            target_update_ms: 목표 사이클 시간(ms) — Phase 4 요구사항 50ms
            timing_window: 사이클 타이밍 이동 평균 윈도우
        """
        self._lock = threading.Lock()
        self._sensors: Dict[str, SensorState] = {}
        self._sensor_char_idx = 0
        self._sensor_char_map: Dict[str, int] = {}  # sensor_id → char index

        # 성능 계측
        self._update_ms_history: deque = deque(maxlen=timing_window)
        self._target_update_ms = target_update_ms
        self._adaptive_relinearize = adaptive_relinearize
        self._current_relinearize_skip = relinearize_skip

        if not GTSAM_AVAILABLE:
            logger.error("GTSAM is not installed. PoseGraphManager in stub mode.")
            self._isam = None
            self._graph = None
            self._initial_values = None
            return

        # iSAM2 설정
        params = ISAM2Params()
        params.setRelinearizeThreshold(relinearize_threshold)
        params.relinearizeSkip = relinearize_skip

        self._isam = ISAM2(params)
        self._graph = NonlinearFactorGraph()
        self._initial_values = Values()

        # 통계
        self._factor_count = 0
        self._update_count = 0
        self._total_variables = 0

        logger.info(
            f"PoseGraphManager initialized: "
            f"relinearize_threshold={relinearize_threshold}, "
            f"relinearize_skip={relinearize_skip}, "
            f"adaptive={adaptive_relinearize}, "
            f"target={target_update_ms}ms"
        )

    # ─────────────────────────────────────────────────────────
    # Symbol Key 관리
    # ─────────────────────────────────────────────────────────

    def _get_sensor_char(self, sensor_id: str) -> int:
        """센서 ID에 대한 gtsam.symbol 문자 인덱스 할당"""
        if sensor_id not in self._sensor_char_map:
            if self._sensor_char_idx >= len(self._SENSOR_CHARS):
                raise RuntimeError(f"Too many sensors (max {len(self._SENSOR_CHARS)})")
            self._sensor_char_map[sensor_id] = self._sensor_char_idx
            self._sensor_char_idx += 1
        return self._sensor_char_map[sensor_id]

    def _make_key(self, sensor_id: str, timestep: int) -> int:
        """센서+시간 → gtsam symbol key"""
        char_idx = self._get_sensor_char(sensor_id)
        char = self._SENSOR_CHARS[char_idx]
        return gtsam.symbol(char, timestep)

    def _ensure_sensor(self, sensor_id: str, is_fixed: bool = False) -> SensorState:
        """센서 상태 레코드 생성 (미존재 시)"""
        if sensor_id not in self._sensors:
            self._sensors[sensor_id] = SensorState(
                sensor_id=sensor_id,
                is_fixed=is_fixed,
            )
            logger.info(f"Registered sensor: {sensor_id} (fixed={is_fixed})")
        return self._sensors[sensor_id]

    # ─────────────────────────────────────────────────────────
    # Factor 추가 API
    # ─────────────────────────────────────────────────────────

    def add_fixed_sensor_prior(
        self,
        sensor_id: str,
        pose_4x4: np.ndarray,
        sigma_rot: float = 0.001,     # ~0.06° — 고정 센서는 매우 높은 confidence
        sigma_trans: float = 0.01,    # 1cm
    ):
        """
        고정 센서(CCTV)의 PriorFactor 추가
        
        Phase 1의 SfM+PnP 결과를 직접 주입합니다.
        고정 센서는 시간 불변이므로 단일 노드(timestep=0)만 사용합니다.
        
        Args:
            sensor_id: 센서 식별자 (e.g. 'sensor_fixed_01')
            pose_4x4: 4x4 world←sensor transform
            sigma_rot: 회전 표준편차 (rad)
            sigma_trans: 이동 표준편차 (m)
        """
        if not GTSAM_AVAILABLE:
            return

        with self._lock:
            state = self._ensure_sensor(sensor_id, is_fixed=True)
            key = self._make_key(sensor_id, 0)
            state.fixed_key = key

            pose = numpy_to_pose3(pose_4x4)
            noise = make_diagonal_noise(np.array([
                sigma_rot, sigma_rot, sigma_rot,
                sigma_trans, sigma_trans, sigma_trans
            ]))

            self._graph.add(PriorFactorPose3(key, pose, noise))
            if not self._initial_values.exists(key):
                self._initial_values.insert(key, pose)
                self._total_variables += 1

            self._factor_count += 1
            state.key_map[0] = key

            logger.info(
                f"Added fixed prior: {sensor_id} "
                f"pos=({pose_4x4[0,3]:.3f}, {pose_4x4[1,3]:.3f}, {pose_4x4[2,3]:.3f})"
            )

    def add_odometry(
        self,
        sensor_id: str,
        timestep_from: int,
        timestep_to: int,
        delta_pose_4x4: np.ndarray,
        sigma_rot: float = 0.02,      # ~1.1°
        sigma_trans: float = 0.05,     # 5cm
        initial_guess_4x4: Optional[np.ndarray] = None,
    ):
        """
        이동형 센서의 Odometry BetweenFactor 추가
        
        VIO/Wheel encoder의 상대 변위를 Factor로 추가합니다.
        
        Args:
            sensor_id: 센서 식별자 (e.g. 'robot_01')
            timestep_from: 시작 시점
            timestep_to: 끝 시점
            delta_pose_4x4: 상대 변위 (4x4)
            sigma_rot: 회전 노이즈 (rad)
            sigma_trans: 이동 노이즈 (m)
            initial_guess_4x4: 새 노드의 초기 추정치 (None이면 전파)
        """
        if not GTSAM_AVAILABLE:
            return

        with self._lock:
            state = self._ensure_sensor(sensor_id, is_fixed=False)
            key_from = self._make_key(sensor_id, timestep_from)
            key_to = self._make_key(sensor_id, timestep_to)

            delta = numpy_to_pose3(delta_pose_4x4)
            noise = make_diagonal_noise(np.array([
                sigma_rot, sigma_rot, sigma_rot,
                sigma_trans, sigma_trans, sigma_trans
            ]))

            self._graph.add(BetweenFactorPose3(key_from, key_to, delta, noise))
            self._factor_count += 1

            # 새 노드 초기값 추가
            if not self._initial_values.exists(key_to):
                if initial_guess_4x4 is not None:
                    guess = numpy_to_pose3(initial_guess_4x4)
                else:
                    # 이전 pose가 있으면 delta를 적용하여 전파
                    try:
                        current_result = self._isam.calculateEstimate()
                        if current_result.exists(key_from):
                            prev_pose = current_result.atPose3(key_from)
                            guess = prev_pose.compose(delta)
                        else:
                            guess = delta  # fallback
                    except Exception:
                        guess = delta
                self._initial_values.insert(key_to, guess)
                self._total_variables += 1

            state.key_map[timestep_to] = key_to
            state.latest_timestep = max(state.latest_timestep, timestep_to)

            logger.debug(
                f"Added odometry: {sensor_id} t{timestep_from}→t{timestep_to}"
            )

    def add_relocalization_prior(
        self,
        sensor_id: str,
        timestep: int,
        pose_4x4: np.ndarray,
        sigma_rot: float = 0.05,      # ~2.9° — relocalization은 odometry보다 부정확
        sigma_trans: float = 0.10,     # 10cm
    ):
        """
        Relocalization PriorFactor 추가
        
        HLoc 등 Visual Relocalization의 결과를 전역 앵커로 주입합니다.
        Drift 보정의 핵심 factor입니다.
        
        Args:
            sensor_id: 센서 식별자
            timestep: 해당 시점
            pose_4x4: 전역 pose (4x4 world←sensor)
            sigma_rot: 회전 불확실성 (rad)
            sigma_trans: 이동 불확실성 (m)
        """
        if not GTSAM_AVAILABLE:
            return

        with self._lock:
            state = self._ensure_sensor(sensor_id, is_fixed=False)
            key = self._make_key(sensor_id, timestep)

            pose = numpy_to_pose3(pose_4x4)
            noise = make_diagonal_noise(np.array([
                sigma_rot, sigma_rot, sigma_rot,
                sigma_trans, sigma_trans, sigma_trans
            ]))

            self._graph.add(PriorFactorPose3(key, pose, noise))
            self._factor_count += 1

            if not self._initial_values.exists(key):
                self._initial_values.insert(key, pose)
                self._total_variables += 1

            state.key_map[timestep] = key
            state.latest_timestep = max(state.latest_timestep, timestep)

            logger.info(
                f"Added relocalization prior: {sensor_id} t={timestep} "
                f"pos=({pose_4x4[0,3]:.3f}, {pose_4x4[1,3]:.3f}, {pose_4x4[2,3]:.3f})"
            )

    def add_inter_sensor_factor(
        self,
        sensor_id_i: str,
        timestep_i: int,
        sensor_id_j: str,
        timestep_j: int,
        relative_pose_4x4: np.ndarray,
        sigma_rot: float = 0.03,
        sigma_trans: float = 0.05,
    ):
        """
        센서 간 상대 변환 BetweenFactor 추가
        
        Overlapping FOV에서 추정된 센서 쌍의 상대 pose를
        Factor Graph에 constraint로 추가합니다.
        
        Args:
            sensor_id_i: 센서 i
            timestep_i: 센서 i 시점
            sensor_id_j: 센서 j
            timestep_j: 센서 j 시점
            relative_pose_4x4: T_i←j (4x4)
            sigma_rot: 회전 노이즈 (rad)
            sigma_trans: 이동 노이즈 (m)
        """
        if not GTSAM_AVAILABLE:
            return

        with self._lock:
            key_i = self._make_key(sensor_id_i, timestep_i)
            key_j = self._make_key(sensor_id_j, timestep_j)

            delta = numpy_to_pose3(relative_pose_4x4)
            noise = make_diagonal_noise(np.array([
                sigma_rot, sigma_rot, sigma_rot,
                sigma_trans, sigma_trans, sigma_trans
            ]))

            self._graph.add(BetweenFactorPose3(key_i, key_j, delta, noise))
            self._factor_count += 1

            logger.info(
                f"Added inter-sensor factor: "
                f"{sensor_id_i}(t{timestep_i}) ↔ {sensor_id_j}(t{timestep_j})"
            )

    def add_gps_prior(
        self,
        sensor_id: str,
        timestep: int,
        position_xyz: np.ndarray,
        sigma_pos: float = 1.0,       # RTK-GPS: ~1m, 후처리: ~2cm
        sigma_rot: float = 1.0,       # GPS는 orientation 제공 안 함 → 매우 큰 불확실성
    ):
        """
        GPS/GNSS PriorFactor (position only, rotation free)
        
        GPS는 위치만 제공하므로 회전 불확실성을 매우 크게 설정합니다.
        """
        if not GTSAM_AVAILABLE:
            return

        with self._lock:
            key = self._make_key(sensor_id, timestep)

            # GPS → position prior만, rotation은 자유
            pose = Pose3(Rot3.Identity(), Point3(*position_xyz))
            noise = make_diagonal_noise(np.array([
                sigma_rot, sigma_rot, sigma_rot,  # rotation free
                sigma_pos, sigma_pos, sigma_pos,
            ]))

            self._graph.add(PriorFactorPose3(key, pose, noise))
            self._factor_count += 1

            if not self._initial_values.exists(key):
                self._initial_values.insert(key, pose)
                self._total_variables += 1

            logger.info(
                f"Added GPS prior: {sensor_id} t={timestep} "
                f"pos=({position_xyz[0]:.3f}, {position_xyz[1]:.3f}, {position_xyz[2]:.3f}) "
                f"σ={sigma_pos:.2f}m"
            )

    def add_uwb_prior(
        self,
        sensor_id: str,
        timestep: int,
        position_xyz: np.ndarray,
        sigma_pos: float = 0.10,      # UWB: ±10cm typical (DW1000 등)
        sigma_rot: float = 1.0,       # UWB도 orientation 제공 안 함
    ):
        """
        UWB Anchor PriorFactor (position only, rotation free)

        실내 환경에서 GPS 불가능 시 UWB 앵커 기반 절대 위치 제공.
        GPS보다 정확도가 높지만(±10cm) 작동 범위가 제한됩니다.

        Args:
            sensor_id: 센서 식별자
            timestep: 해당 시점
            position_xyz: world 좌표계 위치 (m)
            sigma_pos: 위치 표준편차 (m), 기본 10cm
            sigma_rot: 회전 표준편차 (rad), 매우 큼 (rotation free)
        """
        if not GTSAM_AVAILABLE:
            return

        with self._lock:
            state = self._ensure_sensor(sensor_id, is_fixed=False)
            key = self._make_key(sensor_id, timestep)

            pose = Pose3(Rot3.Identity(), Point3(*position_xyz))
            noise = make_diagonal_noise(np.array([
                sigma_rot, sigma_rot, sigma_rot,
                sigma_pos, sigma_pos, sigma_pos,
            ]))

            self._graph.add(PriorFactorPose3(key, pose, noise))
            self._factor_count += 1

            if not self._initial_values.exists(key):
                self._initial_values.insert(key, pose)
                self._total_variables += 1

            state.key_map[timestep] = key
            state.latest_timestep = max(state.latest_timestep, timestep)

            logger.info(
                f"Added UWB prior: {sensor_id} t={timestep} "
                f"pos=({position_xyz[0]:.3f}, {position_xyz[1]:.3f}, {position_xyz[2]:.3f}) "
                f"σ={sigma_pos:.3f}m"
            )

    # ─────────────────────────────────────────────────────────
    # 최적화
    # ─────────────────────────────────────────────────────────

    def optimize(self) -> bool:
        """
        iSAM2 incremental update 수행

        Phase 4: 사이클 타이밍 계측 및 adaptive relinearize_skip 적용

        Returns:
            True if successful, False otherwise
        """
        if not GTSAM_AVAILABLE:
            return False

        with self._lock:
            if self._graph.size() == 0 and self._initial_values.size() == 0:
                return True  # nothing to update

            start_ns = time.perf_counter_ns()
            try:
                self._isam.update(self._graph, self._initial_values)
                # 추가 반복으로 수렴 향상 (선택)
                self._isam.update()

                self._update_count += 1

                # 누적 그래프와 초기값 클리어 (이미 iSAM2에 흡수됨)
                self._graph = NonlinearFactorGraph()
                self._initial_values = Values()

                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                self._update_ms_history.append(elapsed_ms)

                if self._adaptive_relinearize:
                    self._adjust_relinearize_skip(elapsed_ms)

                logger.debug(
                    f"iSAM2 update #{self._update_count}: "
                    f"{self._factor_count} factors, "
                    f"{self._total_variables} vars, "
                    f"{elapsed_ms:.2f}ms"
                )
                return True
            except Exception as e:
                logger.error(f"iSAM2 update failed: {e}")
                return False

    def _adjust_relinearize_skip(self, elapsed_ms: float):
        """사이클 시간 기반 relinearize_skip 자동 조정 (Phase 4)"""
        # 목표 대비 1.5배 이상이면 skip 증가, 0.5배 이하이면 감소
        skip = self._current_relinearize_skip
        if elapsed_ms > self._target_update_ms * 1.5:
            new_skip = min(skip * 2, 200)
        elif elapsed_ms < self._target_update_ms * 0.5 and skip > 1:
            new_skip = max(skip // 2, 1)
        else:
            return

        if new_skip != skip:
            self._current_relinearize_skip = new_skip
            try:
                params = self._isam.params()
                params.relinearizeSkip = new_skip
                logger.info(
                    f"Adaptive relinearize_skip: {skip} → {new_skip} "
                    f"(last cycle {elapsed_ms:.1f}ms vs target {self._target_update_ms}ms)"
                )
            except Exception as e:
                logger.debug(f"Failed to adjust relinearize_skip: {e}")

    # ─────────────────────────────────────────────────────────
    # 결과 조회
    # ─────────────────────────────────────────────────────────

    def get_current_pose(
        self, sensor_id: str, timestep: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        센서의 현재 최적화된 pose 조회
        
        Args:
            sensor_id: 센서 식별자
            timestep: 시점 (None이면 latest)
            
        Returns:
            4x4 numpy transform (world←sensor) or None
        """
        if not GTSAM_AVAILABLE:
            return None

        with self._lock:
            state = self._sensors.get(sensor_id)
            if state is None:
                return None

            if timestep is None:
                if state.is_fixed:
                    timestep = 0
                else:
                    timestep = state.latest_timestep

            key = self._make_key(sensor_id, timestep)
            try:
                result = self._isam.calculateEstimate()
                if result.exists(key):
                    pose = result.atPose3(key)
                    return pose3_to_numpy(pose)
            except Exception as e:
                logger.error(f"Failed to get pose for {sensor_id} t={timestep}: {e}")
            return None

    def get_current_pose_quat(
        self, sensor_id: str, timestep: Optional[int] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        센서의 현재 최적화된 pose를 (quaternion_wxyz, position_xyz)로 반환
        """
        if not GTSAM_AVAILABLE:
            return None

        with self._lock:
            state = self._sensors.get(sensor_id)
            if state is None:
                return None

            if timestep is None:
                if state.is_fixed:
                    timestep = 0
                else:
                    timestep = state.latest_timestep

            key = self._make_key(sensor_id, timestep)
            try:
                result = self._isam.calculateEstimate()
                if result.exists(key):
                    pose = result.atPose3(key)
                    return pose3_to_quat_pos(pose)
            except Exception as e:
                logger.error(f"Failed to get pose for {sensor_id} t={timestep}: {e}")
            return None

    def get_all_current_poses(self) -> Dict[str, np.ndarray]:
        """모든 센서의 latest pose 조회"""
        poses = {}
        for sensor_id in self._sensors:
            pose = self.get_current_pose(sensor_id)
            if pose is not None:
                poses[sensor_id] = pose
        return poses

    def get_marginal_covariance(
        self, sensor_id: str, timestep: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        센서 pose의 주변 공분산(Marginal Covariance) 조회
        
        Returns:
            6x6 covariance matrix (rx, ry, rz, tx, ty, tz)
        """
        if not GTSAM_AVAILABLE:
            return None

        with self._lock:
            state = self._sensors.get(sensor_id)
            if state is None:
                return None

            if timestep is None:
                timestep = 0 if state.is_fixed else state.latest_timestep

            key = self._make_key(sensor_id, timestep)
            try:
                marginals = self._isam.marginalCovariance(key)
                return marginals
            except Exception as e:
                logger.debug(f"Marginal covariance unavailable for {sensor_id}: {e}")
                return None

    # ─────────────────────────────────────────────────────────
    # 유틸리티
    # ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """그래프 통계 반환 (Phase 4: 사이클 타이밍 포함)"""
        timing = self.get_timing_stats()
        return {
            "num_sensors": len(self._sensors),
            "num_variables": self._total_variables,
            "num_factors": self._factor_count,
            "num_updates": self._update_count,
            "timing_ms": timing,
            "relinearize_skip": self._current_relinearize_skip,
            "sensors": {
                sid: {
                    "is_fixed": s.is_fixed,
                    "latest_timestep": s.latest_timestep,
                    "num_keys": len(s.key_map),
                }
                for sid, s in self._sensors.items()
            },
        }

    def get_timing_stats(self) -> dict:
        """사이클 타이밍 통계 (Phase 4 성능 모니터링)"""
        if not self._update_ms_history:
            return {"last": 0.0, "avg": 0.0, "p95": 0.0, "max": 0.0, "count": 0}
        history = np.array(self._update_ms_history)
        return {
            "last": float(history[-1]),
            "avg": float(history.mean()),
            "p95": float(np.percentile(history, 95)),
            "max": float(history.max()),
            "count": int(len(history)),
            "target_ms": float(self._target_update_ms),
        }

    def get_last_update_ms(self) -> float:
        """가장 최근 사이클 시간 (ms)"""
        return float(self._update_ms_history[-1]) if self._update_ms_history else 0.0

    def get_sensor_ids(self) -> List[str]:
        """등록된 센서 ID 목록"""
        return list(self._sensors.keys())

    def is_sensor_registered(self, sensor_id: str) -> bool:
        return sensor_id in self._sensors

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"PoseGraphManager("
            f"sensors={stats['num_sensors']}, "
            f"vars={stats['num_variables']}, "
            f"factors={stats['num_factors']}, "
            f"updates={stats['num_updates']})"
        )
