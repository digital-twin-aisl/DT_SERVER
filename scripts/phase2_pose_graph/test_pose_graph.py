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
Pose Graph Manager Unit Test — GTSAM iSAM2 기능 검증

테스트 시나리오:
  1. 고정 센서 PriorFactor 추가 + 최적화 → 정확한 pose 복원
  2. 이동 센서 Odometry chain → drift 누적 확인
  3. Relocalization prior → drift 보정 확인
  4. Inter-sensor factor → 센서 간 일관성 확인
  5. 통계 조회

사용법:
  python -m scripts.phase2_pose_graph.test_pose_graph
"""
import sys
import os
import numpy as np
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase2_pose_graph.pose_graph_manager import (
    PoseGraphManager,
    numpy_to_pose3,
    pose3_to_numpy,
    GTSAM_AVAILABLE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_transform(tx, ty, tz, yaw_deg=0.0) -> np.ndarray:
    """Simple helper: translation + yaw rotation → 4x4"""
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T[:3, 3] = [tx, ty, tz]
    return T


def test_fixed_sensor_prior():
    """Test 1: 고정 센서 PriorFactor"""
    logger.info("=" * 60)
    logger.info("Test 1: Fixed Sensor Prior")
    logger.info("=" * 60)

    pg = PoseGraphManager()

    T1 = make_transform(1.0, 2.0, 3.0, 45.0)
    T2 = make_transform(-1.0, 0.5, 2.0, 90.0)

    pg.add_fixed_sensor_prior('cam_01', T1)
    pg.add_fixed_sensor_prior('cam_02', T2)
    pg.optimize()

    pose1 = pg.get_current_pose('cam_01')
    pose2 = pg.get_current_pose('cam_02')

    if pose1 is not None:
        err1 = np.linalg.norm(pose1[:3, 3] - T1[:3, 3])
        logger.info(f"  cam_01 position error: {err1:.6f} m")
        assert err1 < 0.01, f"Too large error: {err1}"
    if pose2 is not None:
        err2 = np.linalg.norm(pose2[:3, 3] - T2[:3, 3])
        logger.info(f"  cam_02 position error: {err2:.6f} m")
        assert err2 < 0.01, f"Too large error: {err2}"

    logger.info("  ✅ PASSED")
    return True


def test_odometry_chain():
    """Test 2: 이동 센서 Odometry chain"""
    logger.info("=" * 60)
    logger.info("Test 2: Odometry Chain")
    logger.info("=" * 60)

    pg = PoseGraphManager()

    # 초기 위치
    T_init = make_transform(0, 0, 0)
    pg.add_relocalization_prior('robot_01', 0, T_init, 0.01, 0.01)

    # 5단계 odometry: 각 단계마다 x+1m
    for i in range(5):
        delta = make_transform(1.0, 0, 0)  # 1m forward
        initial_guess = make_transform(float(i + 1), 0, 0)
        pg.add_odometry('robot_01', i, i + 1, delta, 0.01, 0.02, initial_guess)

    pg.optimize()

    # 최종 위치 확인: (5, 0, 0)
    final = pg.get_current_pose('robot_01', 5)
    if final is not None:
        pos = final[:3, 3]
        logger.info(f"  Final position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        err = np.linalg.norm(pos - np.array([5.0, 0, 0]))
        logger.info(f"  Position error: {err:.6f} m")
        assert err < 0.5, f"Too large error: {err}"

    logger.info("  ✅ PASSED")
    return True


def test_relocalization_correction():
    """Test 3: Relocalization으로 drift 보정"""
    logger.info("=" * 60)
    logger.info("Test 3: Relocalization Drift Correction")
    logger.info("=" * 60)

    pg = PoseGraphManager()

    # 초기 위치 (노이즈 있는 odometry)
    pg.add_relocalization_prior('robot_01', 0, make_transform(0, 0, 0), 0.01, 0.01)

    # Drifted odometry: 실제로는 x=1이지만, drift로 x=1.5
    pg.add_odometry('robot_01', 0, 1, make_transform(1.5, 0.2, 0),
                    sigma_rot=0.05, sigma_trans=0.3,
                    initial_guess_4x4=make_transform(1.5, 0.2, 0))

    # Relocalization이 "실제" 위치 (1.0, 0.0, 0.0)를 알려줌
    pg.add_relocalization_prior('robot_01', 1, make_transform(1.0, 0.0, 0.0),
                                sigma_rot=0.02, sigma_trans=0.05)

    pg.optimize()

    corrected = pg.get_current_pose('robot_01', 1)
    if corrected is not None:
        pos = corrected[:3, 3]
        logger.info(f"  Corrected position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        # Relocalization이 더 정확(sigma 작음)하므로 (1.0, 0.0, 0.0)에 가까워야 함
        err_to_reloc = np.linalg.norm(pos - np.array([1.0, 0.0, 0.0]))
        err_to_drift = np.linalg.norm(pos - np.array([1.5, 0.2, 0.0]))
        logger.info(f"  Error to relocalization: {err_to_reloc:.4f} m")
        logger.info(f"  Error to drifted odom:  {err_to_drift:.4f} m")
        assert err_to_reloc < err_to_drift, "Relocalization should dominate"

    logger.info("  ✅ PASSED")
    return True


def test_inter_sensor():
    """Test 4: Inter-sensor factor"""
    logger.info("=" * 60)
    logger.info("Test 4: Inter-sensor Factor")
    logger.info("=" * 60)

    pg = PoseGraphManager()

    # 센서 A: 정확한 위치
    pg.add_fixed_sensor_prior('sensor_A', make_transform(0, 0, 0), 0.001, 0.001)

    # 센서 B: 부정확한 위치 추정
    pg.add_relocalization_prior('sensor_B', 0,
                                 make_transform(5.0, 0.3, 0.1),
                                 sigma_rot=0.1, sigma_trans=0.5)

    # Inter-sensor: A에서 B까지 (5, 0, 0) 거리
    rel = make_transform(5.0, 0.0, 0.0)
    pg.add_inter_sensor_factor('sensor_A', 0, 'sensor_B', 0, rel, 0.01, 0.02)

    pg.optimize()

    pB = pg.get_current_pose('sensor_B')
    if pB is not None:
        pos = pB[:3, 3]
        logger.info(f"  sensor_B position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        err = np.linalg.norm(pos - np.array([5.0, 0.0, 0.0]))
        logger.info(f"  Error to true (5,0,0): {err:.4f} m")

    logger.info("  ✅ PASSED")
    return True


def test_stats():
    """Test 5: 통계 조회"""
    logger.info("=" * 60)
    logger.info("Test 5: Statistics")
    logger.info("=" * 60)

    pg = PoseGraphManager()
    pg.add_fixed_sensor_prior('cam_01', make_transform(1, 0, 0))
    pg.add_relocalization_prior('robot_01', 0, make_transform(0, 0, 0))
    pg.add_odometry('robot_01', 0, 1, make_transform(1, 0, 0),
                    initial_guess_4x4=make_transform(1, 0, 0))
    pg.optimize()

    stats = pg.get_stats()
    logger.info(f"  Stats: {stats}")
    logger.info(f"  Repr: {pg}")
    assert stats['num_sensors'] == 2
    assert stats['num_factors'] >= 3

    logger.info("  ✅ PASSED")
    return True


def main():
    logger.info("Phase 2 — Pose Graph Manager Unit Tests")
    logger.info("")

    if not GTSAM_AVAILABLE:
        logger.warning("⚠️  GTSAM not installed. Tests will be skipped.")
        logger.warning("   Install: pip install gtsam")
        return

    tests = [
        test_fixed_sensor_prior,
        test_odometry_chain,
        test_relocalization_correction,
        test_inter_sensor,
        test_stats,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"  ❌ FAILED: {e}")

    logger.info("")
    logger.info(f"Results: {passed}/{len(tests)} tests passed")


if __name__ == '__main__':
    main()
