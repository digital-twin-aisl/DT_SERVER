"""
Submap Manager — 분산 Pose Graph (서브맵 분할)

대규모 환경(수백 대 센서, 장기 운용)에서 단일 거대 그래프는
iSAM2도 O(k log N) 한계에 부딪힙니다. 센서/공간 단위로
**submap**을 분할하여 각각 독립적인 PoseGraphManager로 운영하고,
submap 경계를 가로지르는 inter-sensor factor는 별도로 관리합니다.

설계:
  - 각 submap은 자체 PoseGraphManager (=iSAM2) 보유
  - sensor_id → submap_id 매핑 테이블
  - cross-submap factor는 별도 큐에 저장 (현재는 통계 목적)
    실제 글로벌 정합은 향후 anchor sync 단계에서 처리
  - 모든 라우팅은 SubmapManager가 담당하므로
    상위 노드는 단일 인터페이스로 사용 가능

사용법:
    sm = SubmapManager(default_submap='global')
    sm.assign_sensor('robot_01', 'zone_A')
    sm.assign_sensor('cam_fixed_01', 'zone_A')
    sm.add_relocalization_prior('robot_01', 0, T, 0.05, 0.10)
    sm.add_inter_sensor_factor('robot_01', 0, 'cam_fixed_01', 0, T_rel)
    sm.optimize_all()
    pose = sm.get_current_pose('robot_01')
"""
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import threading

from scripts.phase2_pose_graph.pose_graph_manager import (
    PoseGraphManager, GTSAM_AVAILABLE,
)

logger = logging.getLogger(__name__)


@dataclass
class CrossSubmapFactor:
    """서로 다른 submap에 속한 두 센서 사이의 inter-sensor factor"""
    sensor_id_i: str
    submap_i: str
    timestep_i: int
    sensor_id_j: str
    submap_j: str
    timestep_j: int
    relative_pose_4x4: np.ndarray
    sigma_rot: float
    sigma_trans: float


class SubmapManager:
    """
    분산 Pose Graph 관리자

    여러 PoseGraphManager 인스턴스를 묶어 sensor → submap 라우팅을
    제공하고, cross-submap factor를 별도 추적합니다.
    """

    def __init__(
        self,
        default_submap: str = 'global',
        relinearize_threshold: float = 0.1,
        relinearize_skip: int = 10,
        adaptive_relinearize: bool = True,
        target_update_ms: float = 50.0,
    ):
        """
        Args:
            default_submap: 명시 할당 없는 센서의 기본 submap
            relinearize_threshold/skip/adaptive: 각 submap PoseGraphManager 인자
            target_update_ms: Phase 4 목표 사이클 시간 (50ms)
        """
        self._lock = threading.Lock()
        self._default_submap = default_submap
        self._pg_kwargs = dict(
            relinearize_threshold=relinearize_threshold,
            relinearize_skip=relinearize_skip,
            adaptive_relinearize=adaptive_relinearize,
            target_update_ms=target_update_ms,
        )

        # submap_id → PoseGraphManager
        self._submaps: Dict[str, PoseGraphManager] = {}
        # sensor_id → submap_id
        self._sensor_to_submap: Dict[str, str] = {}
        # cross-submap factor 큐 (감사/통계용)
        self._cross_factors: deque = deque(maxlen=1000)
        self._cross_factor_count = 0

        # 기본 submap 미리 생성
        self._get_or_create_submap(default_submap)

        logger.info(
            f"SubmapManager initialized: default='{default_submap}', "
            f"target={target_update_ms}ms"
        )

    # ─────────────────────────────────────────────────────────
    # Submap / Sensor 관리
    # ─────────────────────────────────────────────────────────

    def _get_or_create_submap(self, submap_id: str) -> PoseGraphManager:
        """submap 인스턴스 조회/생성"""
        if submap_id not in self._submaps:
            self._submaps[submap_id] = PoseGraphManager(**self._pg_kwargs)
            logger.info(f"Created submap: '{submap_id}'")
        return self._submaps[submap_id]

    def assign_sensor(self, sensor_id: str, submap_id: str) -> None:
        """센서를 특정 submap에 할당"""
        with self._lock:
            prev = self._sensor_to_submap.get(sensor_id)
            if prev is not None and prev != submap_id:
                logger.warning(
                    f"Sensor '{sensor_id}' reassigned: {prev} → {submap_id}. "
                    f"기존 submap의 factor는 그대로 유지됩니다."
                )
            self._sensor_to_submap[sensor_id] = submap_id
            self._get_or_create_submap(submap_id)

    def get_submap_for_sensor(self, sensor_id: str) -> str:
        """센서의 submap_id 조회 (미할당 시 default)"""
        return self._sensor_to_submap.get(sensor_id, self._default_submap)

    def get_submap(self, submap_id: str) -> Optional[PoseGraphManager]:
        """submap의 PoseGraphManager 직접 접근"""
        return self._submaps.get(submap_id)

    def list_submaps(self) -> List[str]:
        return list(self._submaps.keys())

    # ─────────────────────────────────────────────────────────
    # Factor 라우팅 API (PoseGraphManager 인터페이스 미러)
    # ─────────────────────────────────────────────────────────

    def add_fixed_sensor_prior(
        self, sensor_id: str, pose_4x4: np.ndarray,
        sigma_rot: float = 0.001, sigma_trans: float = 0.01,
    ):
        sm = self._get_or_create_submap(self.get_submap_for_sensor(sensor_id))
        sm.add_fixed_sensor_prior(sensor_id, pose_4x4, sigma_rot, sigma_trans)

    def add_odometry(
        self, sensor_id: str, timestep_from: int, timestep_to: int,
        delta_pose_4x4: np.ndarray,
        sigma_rot: float = 0.02, sigma_trans: float = 0.05,
        initial_guess_4x4: Optional[np.ndarray] = None,
    ):
        sm = self._get_or_create_submap(self.get_submap_for_sensor(sensor_id))
        sm.add_odometry(
            sensor_id, timestep_from, timestep_to, delta_pose_4x4,
            sigma_rot, sigma_trans, initial_guess_4x4,
        )

    def add_relocalization_prior(
        self, sensor_id: str, timestep: int, pose_4x4: np.ndarray,
        sigma_rot: float = 0.05, sigma_trans: float = 0.10,
    ):
        sm = self._get_or_create_submap(self.get_submap_for_sensor(sensor_id))
        sm.add_relocalization_prior(sensor_id, timestep, pose_4x4, sigma_rot, sigma_trans)

    def add_gps_prior(
        self, sensor_id: str, timestep: int, position_xyz: np.ndarray,
        sigma_pos: float = 1.0, sigma_rot: float = 1.0,
    ):
        sm = self._get_or_create_submap(self.get_submap_for_sensor(sensor_id))
        sm.add_gps_prior(sensor_id, timestep, position_xyz, sigma_pos, sigma_rot)

    def add_uwb_prior(
        self, sensor_id: str, timestep: int, position_xyz: np.ndarray,
        sigma_pos: float = 0.10, sigma_rot: float = 1.0,
    ):
        sm = self._get_or_create_submap(self.get_submap_for_sensor(sensor_id))
        sm.add_uwb_prior(sensor_id, timestep, position_xyz, sigma_pos, sigma_rot)

    def add_inter_sensor_factor(
        self, sensor_id_i: str, timestep_i: int,
        sensor_id_j: str, timestep_j: int,
        relative_pose_4x4: np.ndarray,
        sigma_rot: float = 0.03, sigma_trans: float = 0.05,
    ):
        """
        Inter-sensor factor 라우팅

        두 센서가 같은 submap이면 해당 submap에 BetweenFactor 추가,
        서로 다른 submap이면 cross-submap factor 큐에 저장합니다.
        """
        submap_i = self.get_submap_for_sensor(sensor_id_i)
        submap_j = self.get_submap_for_sensor(sensor_id_j)

        if submap_i == submap_j:
            sm = self._get_or_create_submap(submap_i)
            sm.add_inter_sensor_factor(
                sensor_id_i, timestep_i, sensor_id_j, timestep_j,
                relative_pose_4x4, sigma_rot, sigma_trans,
            )
        else:
            # Cross-submap: 큐에 저장 (현재는 logging+통계 목적)
            cf = CrossSubmapFactor(
                sensor_id_i=sensor_id_i, submap_i=submap_i, timestep_i=timestep_i,
                sensor_id_j=sensor_id_j, submap_j=submap_j, timestep_j=timestep_j,
                relative_pose_4x4=relative_pose_4x4.copy(),
                sigma_rot=sigma_rot, sigma_trans=sigma_trans,
            )
            with self._lock:
                self._cross_factors.append(cf)
                self._cross_factor_count += 1
            logger.info(
                f"Cross-submap factor queued: "
                f"{sensor_id_i}[{submap_i}](t{timestep_i}) ↔ "
                f"{sensor_id_j}[{submap_j}](t{timestep_j}) "
                f"(total cross={self._cross_factor_count})"
            )

    # ─────────────────────────────────────────────────────────
    # 최적화 & 조회
    # ─────────────────────────────────────────────────────────

    def optimize_all(self) -> Dict[str, bool]:
        """모든 submap을 순차 최적화"""
        results = {}
        for sid, sm in self._submaps.items():
            results[sid] = sm.optimize()
        return results

    def get_current_pose(
        self, sensor_id: str, timestep: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        sm = self._submaps.get(self.get_submap_for_sensor(sensor_id))
        if sm is None:
            return None
        return sm.get_current_pose(sensor_id, timestep)

    def get_all_current_poses(self) -> Dict[str, np.ndarray]:
        """모든 submap에 걸쳐 등록된 센서의 latest pose 통합"""
        poses: Dict[str, np.ndarray] = {}
        for sm in self._submaps.values():
            poses.update(sm.get_all_current_poses())
        return poses

    def iter_sensors(self):
        """
        모든 submap의 등록된 센서 순회

        Yields:
            (sensor_id, is_fixed, latest_timestep, submap_id)
        """
        for submap_id, sm in self._submaps.items():
            for sid, state in sm._sensors.items():
                yield sid, state.is_fixed, state.latest_timestep, submap_id

    def get_stats(self) -> dict:
        """전체 분산 그래프 통계"""
        per_submap = {sid: sm.get_stats() for sid, sm in self._submaps.items()}
        total_factors = sum(s['num_factors'] for s in per_submap.values())
        total_vars = sum(s['num_variables'] for s in per_submap.values())
        timings = [s['timing_ms']['last'] for s in per_submap.values()
                   if s['timing_ms']['count'] > 0]
        return {
            'num_submaps': len(self._submaps),
            'num_sensors_assigned': len(self._sensor_to_submap),
            'total_factors': total_factors,
            'total_variables': total_vars,
            'cross_submap_factors': self._cross_factor_count,
            'max_last_cycle_ms': max(timings) if timings else 0.0,
            'per_submap': per_submap,
        }

    def get_cross_submap_factors(self) -> List[CrossSubmapFactor]:
        """현재까지 누적된 cross-submap factor 스냅샷"""
        with self._lock:
            return list(self._cross_factors)

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"SubmapManager(submaps={s['num_submaps']}, "
            f"sensors={s['num_sensors_assigned']}, "
            f"factors={s['total_factors']}, "
            f"cross={s['cross_submap_factors']})"
        )
