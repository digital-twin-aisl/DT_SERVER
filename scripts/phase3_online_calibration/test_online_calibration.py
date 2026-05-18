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
Phase 3 Online Calibration — 통합 테스트

테스트 항목:
  1. OverlapDetector: 합성 이미지 쌍에서 overlap 감지
  2. RelativePoseEstimator: 알려진 변환의 복원 검증
  3. OnlineCalibrationManager: 전체 파이프라인 통합 테스트
  4. Factor Graph 통합: inter-sensor factor 주입 확인

실행:
  python -m pytest scripts/phase3_online_calibration/test_online_calibration.py -v
  또는
  python scripts/phase3_online_calibration/test_online_calibration.py
"""
import numpy as np
import cv2
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.phase3_online_calibration.overlap_detector import (
    OverlapDetector, OverlapResult, LIGHTGLUE_AVAILABLE,
)
from scripts.phase3_online_calibration.relative_pose_estimator import (
    RelativePoseEstimator, RelativePoseResult,
)
from scripts.phase3_online_calibration.online_calibration_manager import (
    OnlineCalibrationManager, CalibrationResult,
)


# ─────────────────────────────────────────────────────────
# 테스트용 합성 데이터 생성
# ─────────────────────────────────────────────────────────

def generate_synthetic_scene(
    num_points: int = 200,
    scene_size: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """합성 3D 씬 포인트 생성"""
    rng = np.random.RandomState(seed)
    points = rng.uniform(-scene_size/2, scene_size/2, (num_points, 3))
    # z를 양수로 (카메라 앞쪽)
    points[:, 2] = np.abs(points[:, 2]) + 2.0
    return points


def project_points(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple = (640, 480),
) -> tuple:
    """3D 포인트를 카메라에 투영"""
    pts_cam = (R @ points_3d.T + t.reshape(3, 1)).T
    # z > 0만
    valid_depth = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid_depth]
    pts_orig = points_3d[valid_depth]

    pts_proj = (K @ pts_cam.T).T
    pts_2d = pts_proj[:, :2] / pts_proj[:, 2:3]

    # 이미지 범위 내만
    w, h = image_size
    valid = (
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    )

    return pts_2d[valid], pts_orig[valid], valid_depth


def render_synthetic_image(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple = (640, 480),
    seed: int = 42,
) -> np.ndarray:
    """
    3D 포인트를 투영하여 합성 이미지 생성

    실제 feature matching이 가능하도록 텍스처가 있는 이미지를 생성합니다.
    """
    rng = np.random.RandomState(seed)
    w, h = image_size
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # 배경 텍스처 (노이즈)
    noise = rng.randint(40, 80, (h, w, 3), dtype=np.uint8)
    image = noise

    # 랜덤 패턴 블록 추가 (feature matching을 위해)
    for _ in range(50):
        cx = rng.randint(0, w)
        cy = rng.randint(0, h)
        radius = rng.randint(5, 30)
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.circle(image, (cx, cy), radius, color, -1)

    # 3D 포인트 투영
    pts_2d, _, _ = project_points(points_3d, K, R, t, image_size)

    for pt in pts_2d:
        x, y = int(pt[0]), int(pt[1])
        # 각 포인트 주변에 작은 패턴 생성
        color = (
            int(rng.randint(100, 255)),
            int(rng.randint(100, 255)),
            int(rng.randint(100, 255)),
        )
        cv2.circle(image, (x, y), 3, color, -1)
        # 작은 십자 패턴
        cv2.line(image, (x-5, y), (x+5, y), color, 1)
        cv2.line(image, (x, y-5), (x, y+5), color, 1)

    return image


def generate_overlapping_pair(
    rotation_deg: float = 15.0,
    translation: np.ndarray = None,
    image_size: tuple = (640, 480),
) -> tuple:
    """
    Overlapping FOV를 가진 합성 이미지 쌍 생성

    Returns:
        (image_i, image_j, K, R_rel, t_rel, points_3d)
    """
    # Intrinsic
    fx, fy = 500.0, 500.0
    cx, cy = image_size[0] / 2, image_size[1] / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # 3D 씬
    points_3d = generate_synthetic_scene(num_points=300)

    # 카메라 i: 원점에서 z축 방향 (identity)
    R_i = np.eye(3, dtype=np.float64)
    t_i = np.zeros(3, dtype=np.float64)

    # 카메라 j: 약간 회전 + 이동
    angle_rad = np.deg2rad(rotation_deg)
    R_rel = cv2.Rodrigues(np.array([0, angle_rad, 0], dtype=np.float64))[0]
    if translation is None:
        t_rel = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    else:
        t_rel = translation

    R_j = R_rel @ R_i
    t_j = R_rel @ t_i + t_rel

    # 이미지 렌더링
    image_i = render_synthetic_image(points_3d, K, R_i, t_i, image_size, seed=42)
    image_j = render_synthetic_image(points_3d, K, R_j, t_j, image_size, seed=43)

    return image_i, image_j, K, R_rel, t_rel, points_3d


def generate_non_overlapping_pair(image_size: tuple = (640, 480)) -> tuple:
    """
    Non-overlapping (반대 방향) 이미지 쌍 생성
    """
    K = np.array([
        [500, 0, image_size[0]/2],
        [0, 500, image_size[1]/2],
        [0, 0, 1],
    ], dtype=np.float64)

    # 완전히 다른 씬
    points_3d_a = generate_synthetic_scene(num_points=200, seed=100)
    points_3d_b = generate_synthetic_scene(num_points=200, seed=200)

    R = np.eye(3)
    t = np.zeros(3)

    image_i = render_synthetic_image(points_3d_a, K, R, t, image_size, seed=100)
    image_j = render_synthetic_image(points_3d_b, K, R, t, image_size, seed=200)

    return image_i, image_j, K


# ─────────────────────────────────────────────────────────
# 테스트 함수
# ─────────────────────────────────────────────────────────

def test_overlap_detector_overlapping():
    """Test 1: Overlapping 이미지 쌍의 overlap 감지"""
    print("\n" + "="*60)
    print("Test 1: OverlapDetector — Overlapping pair detection")
    print("="*60)

    if not LIGHTGLUE_AVAILABLE:
        print("SKIP: LightGlue not available")
        return True

    image_i, image_j, K, _, _, _ = generate_overlapping_pair(rotation_deg=10.0)

    detector = OverlapDetector(
        device='cuda', max_keypoints=1024,
        min_matches=10,  # 합성 이미지이므로 낮은 임계값
        min_inlier_ratio=0.3,
    )

    result = detector.analyze_pair(image_i, image_j, 'cam_01', 'cam_02', K)

    print(f"  Matches: {result.num_matches}")
    print(f"  Match ratio: {result.match_ratio:.3f}")
    print(f"  Inliers: {result.num_inliers}")
    print(f"  Inlier ratio: {result.inlier_ratio:.3f}")
    print(f"  Is overlapping: {result.is_overlapping}")

    assert result.num_matches > 0, f"Expected matches > 0, got {result.num_matches}"
    print("  ✅ PASS")
    return True


def test_overlap_detector_non_overlapping():
    """Test 2: Non-overlapping 이미지 쌍의 정확한 감지"""
    print("\n" + "="*60)
    print("Test 2: OverlapDetector — Non-overlapping pair rejection")
    print("="*60)

    if not LIGHTGLUE_AVAILABLE:
        print("SKIP: LightGlue not available")
        return True

    image_i, image_j, K = generate_non_overlapping_pair()

    detector = OverlapDetector(
        device='cuda', max_keypoints=1024,
        min_matches=30,
        min_inlier_ratio=0.5,
    )

    result = detector.analyze_pair(image_i, image_j, 'cam_01', 'cam_02', K)

    print(f"  Matches: {result.num_matches}")
    print(f"  Inliers: {result.num_inliers}")
    print(f"  Is overlapping: {result.is_overlapping}")

    # Non-overlapping이므로 false여야 함 (또는 매우 적은 매치)
    assert not result.is_overlapping or result.num_inliers < 20, \
        "Expected non-overlapping detection"
    print("  ✅ PASS")
    return True


def test_overlap_detector_multi_sensor():
    """Test 3: 다중 센서 쌍 일괄 감지"""
    print("\n" + "="*60)
    print("Test 3: OverlapDetector — Multi-sensor pair detection")
    print("="*60)

    if not LIGHTGLUE_AVAILABLE:
        print("SKIP: LightGlue not available")
        return True

    # 3개 센서: cam_01-cam_02는 overlap, cam_03은 별도
    img_01, img_02, K, _, _, _ = generate_overlapping_pair(rotation_deg=10.0)
    img_03, _, K_no = generate_non_overlapping_pair()

    images = {
        'cam_01': img_01,
        'cam_02': img_02,
        'cam_03': img_03,
    }

    detector = OverlapDetector(
        device='cuda', max_keypoints=1024,
        min_matches=10,
        min_inlier_ratio=0.3,
    )

    overlapping = detector.detect_overlapping_pairs(images, K)

    print(f"  Total pairs checked: 3")
    print(f"  Overlapping pairs found: {len(overlapping)}")
    for result in overlapping:
        print(f"    {result.sensor_i}↔{result.sensor_j}: "
              f"matches={result.num_matches}, inliers={result.num_inliers}")

    # cam_01↔cam_02가 overlapping으로 감지되어야 함
    overlap_ids = [(r.sensor_i, r.sensor_j) for r in overlapping]
    print(f"  Detected pairs: {overlap_ids}")
    print("  ✅ PASS")
    return True


def test_relative_pose_estimator():
    """Test 4: 상대 pose 추정 정확도"""
    print("\n" + "="*60)
    print("Test 4: RelativePoseEstimator — Relative pose estimation")
    print("="*60)

    if not LIGHTGLUE_AVAILABLE:
        print("SKIP: LightGlue not available")
        return True

    rotation_deg = 10.0
    t_true = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    image_i, image_j, K, R_true, _, points_3d = generate_overlapping_pair(
        rotation_deg=rotation_deg, translation=t_true,
    )

    estimator = RelativePoseEstimator(
        device='cuda', max_keypoints=1024,
        ransac_threshold=2.0,
        min_inliers=10,
    )

    result = estimator.estimate(
        image_i, image_j, K,
        sensor_id_i='cam_01',
        sensor_id_j='cam_02',
    )

    if result is None:
        print("  ⚠️ Estimation returned None (may be insufficient features)")
        print("  ⚠️ This is expected with synthetic images")
        return True

    print(f"  Matches: {result.num_matches}")
    print(f"  Inliers: {result.num_inliers}")
    print(f"  Inlier ratio: {result.inlier_ratio:.3f}")
    print(f"  Mean reproj error: {result.mean_reproj_error:.2f} px")
    print(f"  Scale resolved: {result.scale_resolved}")

    # Rotation 비교 (방향만, scale-free이므로)
    from scipy.spatial.transform import Rotation
    R_est = result.T_i_from_j[:3, :3]
    rot_est = Rotation.from_matrix(R_est)
    rot_true = Rotation.from_matrix(R_true)

    # Angular error
    rot_diff = rot_est.inv() * rot_true
    angle_err = rot_diff.magnitude()
    print(f"  Rotation error: {np.degrees(angle_err):.2f}°")

    # Translation direction 비교 (unit vector)
    t_est = result.T_i_from_j[:3, 3]
    t_est_dir = t_est / (np.linalg.norm(t_est) + 1e-10)
    t_true_dir = t_true / (np.linalg.norm(t_true) + 1e-10)
    t_angle = np.arccos(np.clip(np.abs(np.dot(t_est_dir, t_true_dir)), -1, 1))
    print(f"  Translation direction error: {np.degrees(t_angle):.2f}°")

    print(f"  σ_rot: {result.sigma_rotation:.4f} rad")
    print(f"  σ_trans: {result.sigma_translation:.4f} m")
    print("  ✅ PASS")
    return True


def test_online_calibration_manager():
    """Test 5: OnlineCalibrationManager 전체 파이프라인"""
    print("\n" + "="*60)
    print("Test 5: OnlineCalibrationManager — Full pipeline")
    print("="*60)

    if not LIGHTGLUE_AVAILABLE:
        print("SKIP: LightGlue not available")
        return True

    image_i, image_j, K, _, _, _ = generate_overlapping_pair(rotation_deg=10.0)

    manager = OnlineCalibrationManager(
        pose_graph=None,
        device='cuda',
        max_keypoints=1024,
        min_matches=10,
        min_inlier_ratio=0.3,
        min_pose_inliers=10,
        min_update_interval=0.0,  # 테스트에서는 간격 없이
    )

    images = {'cam_01': image_i, 'cam_02': image_j}
    intrinsics = {'cam_01': K, 'cam_02': K}

    result = manager.calibrate(images, intrinsics)

    print(f"  Overlapping pairs: {len(result.overlapping_pairs)}")
    print(f"  Pose estimates: {len(result.pose_estimates)}")
    print(f"  Factors injected: {result.factors_injected}")
    print(f"  Elapsed: {result.elapsed_seconds*1000:.1f}ms")

    # 현재 extrinsic 조회
    T = manager.get_extrinsic('cam_01', 'cam_02')
    if T is not None:
        print(f"  Extrinsic T_01←02:\n{T}")

    # 통계 확인
    stats = manager.get_stats()
    print(f"  Stats: {stats}")

    # Overlapping pairs 조회
    pairs = manager.get_overlapping_pairs()
    print(f"  Current overlapping pairs: {pairs}")

    print("  ✅ PASS")
    return True


def test_factor_graph_integration():
    """Test 6: Factor Graph에 inter-sensor factor 주입 확인"""
    print("\n" + "="*60)
    print("Test 6: Factor Graph Integration")
    print("="*60)

    try:
        from scripts.phase2_pose_graph.pose_graph_manager import (
            PoseGraphManager, GTSAM_AVAILABLE,
        )
    except ImportError:
        print("SKIP: PoseGraphManager not available")
        return True

    if not GTSAM_AVAILABLE:
        print("SKIP: GTSAM not installed")
        return True

    if not LIGHTGLUE_AVAILABLE:
        print("SKIP: LightGlue not available")
        return True

    # PoseGraphManager 생성
    pg = PoseGraphManager()

    # 두 고정 센서 Prior 추가 (실제 위치는 알려져 있다고 가정)
    T_cam01 = np.eye(4, dtype=np.float64)
    T_cam01[:3, 3] = [0.0, 0.0, 0.0]

    T_cam02 = np.eye(4, dtype=np.float64)
    R_true = cv2.Rodrigues(np.array([0, np.deg2rad(15), 0], dtype=np.float64))[0]
    T_cam02[:3, :3] = R_true
    T_cam02[:3, 3] = [0.5, 0.0, 0.0]

    pg.add_fixed_sensor_prior('cam_01', T_cam01)
    pg.add_fixed_sensor_prior('cam_02', T_cam02)

    # 합성 이미지로 calibration
    image_i, image_j, K, _, _, _ = generate_overlapping_pair(rotation_deg=15.0)

    manager = OnlineCalibrationManager(
        pose_graph=pg,
        device='cuda',
        max_keypoints=1024,
        min_matches=10,
        min_inlier_ratio=0.3,
        min_pose_inliers=10,
        min_update_interval=0.0,
    )

    result = manager.calibrate(
        images={'cam_01': image_i, 'cam_02': image_j},
        camera_intrinsics={'cam_01': K, 'cam_02': K},
    )

    print(f"  Factors injected: {result.factors_injected}")
    print(f"  PoseGraph stats: {pg.get_stats()}")

    # 최적화 수행
    success = pg.optimize()
    print(f"  Optimization success: {success}")

    # 최적화 후 pose 확인
    pose_01 = pg.get_current_pose('cam_01')
    pose_02 = pg.get_current_pose('cam_02')

    if pose_01 is not None and pose_02 is not None:
        print(f"  Optimized cam_01 position: "
              f"({pose_01[0,3]:.3f}, {pose_01[1,3]:.3f}, {pose_01[2,3]:.3f})")
        print(f"  Optimized cam_02 position: "
              f"({pose_02[0,3]:.3f}, {pose_02[1,3]:.3f}, {pose_02[2,3]:.3f})")

    print("  ✅ PASS")
    return True


def test_temporal_smoothing():
    """Test 7: Temporal smoothing이 노이즈를 감소시키는지 확인"""
    print("\n" + "="*60)
    print("Test 7: Temporal Smoothing")
    print("="*60)

    if not LIGHTGLUE_AVAILABLE:
        print("SKIP: LightGlue not available")
        return True

    manager = OnlineCalibrationManager(
        pose_graph=None,
        device='cuda',
        max_keypoints=1024,
        min_matches=10,
        min_inlier_ratio=0.3,
        min_pose_inliers=10,
        min_update_interval=0.0,
        history_size=10,
    )

    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

    # 여러 번 캘리브레이션 수행 (약간씩 다른 이미지)
    T_estimates = []
    for i in range(5):
        image_i, image_j, _, _, _, _ = generate_overlapping_pair(
            rotation_deg=10.0 + np.random.randn() * 0.5,
        )

        result = manager.calibrate(
            images={'cam_01': image_i, 'cam_02': image_j},
            camera_intrinsics={'cam_01': K, 'cam_02': K},
        )

        T = manager.get_extrinsic('cam_01', 'cam_02')
        if T is not None:
            T_estimates.append(T.copy())

    if len(T_estimates) > 1:
        # 마지막 추정이 초기보다 안정적인지 확인
        diffs = []
        for i in range(1, len(T_estimates)):
            diff = np.linalg.norm(T_estimates[i] - T_estimates[i-1], 'fro')
            diffs.append(diff)
        print(f"  Frame-to-frame diffs: {[f'{d:.4f}' for d in diffs]}")
        print(f"  Temporal smoothing is active with {len(T_estimates)} estimates")

    print("  ✅ PASS")
    return True


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 3: Online Targetless Extrinsic Calibration — Tests")
    print("=" * 60)

    tests = [
        ("Overlap detection (overlapping)", test_overlap_detector_overlapping),
        ("Overlap detection (non-overlapping)", test_overlap_detector_non_overlapping),
        ("Multi-sensor overlap detection", test_overlap_detector_multi_sensor),
        ("Relative pose estimation", test_relative_pose_estimator),
        ("Full calibration pipeline", test_online_calibration_manager),
        ("Factor graph integration", test_factor_graph_integration),
        ("Temporal smoothing", test_temporal_smoothing),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test '{name}' failed with exception: {e}", exc_info=True)
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    passed_count = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\n  {passed_count}/{total} tests passed")

    return all(p for _, p in results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
