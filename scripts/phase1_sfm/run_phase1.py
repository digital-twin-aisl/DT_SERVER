#!/usr/bin/env python3
"""
Phase 1 실행 스크립트 - SfM 맵 검증 및 고정 카메라 PnP 추정 테스트

사용법:
  python -m scripts.phase1_sfm.run_phase1 --sfm_model_path temp_data/sfm_output/sparse_txt \
      --db_image_dir Collabolab --output_dir temp_data/phase1_results
"""
import sys
import os
import json
import argparse
import logging
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1_sfm.sfm_map import SfMMap
from scripts.phase1_sfm.pnp_localizer import PnPLocalizer, save_poses_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_sfm_map(sfm_map: SfMMap) -> dict:
    """SfM 맵 품질 검증"""
    logger.info("=" * 60)
    logger.info("SfM Map Validation")
    logger.info("=" * 60)

    # Camera info
    for cam_id, cam in sfm_map.cameras.items():
        logger.info(f"Camera {cam_id}: {cam.model} {cam.width}x{cam.height}")
        logger.info(f"  K = {cam.K.tolist()}")
        logger.info(f"  dist = {cam.dist_coeffs.tolist()}")

    # Image stats
    positions = []
    for img_id, img in sfm_map.images.items():
        positions.append(img.position)
    positions = np.array(positions)

    logger.info(f"\nRegistered images: {len(sfm_map.images)}")
    logger.info(f"3D points: {len(sfm_map.points3D)}")
    logger.info(f"Camera positions range:")
    logger.info(f"  X: [{positions[:,0].min():.3f}, {positions[:,0].max():.3f}]")
    logger.info(f"  Y: [{positions[:,1].min():.3f}, {positions[:,1].max():.3f}]")
    logger.info(f"  Z: [{positions[:,2].min():.3f}, {positions[:,2].max():.3f}]")

    # Track length distribution
    track_lens = [len(p.track) for p in sfm_map.points3D.values()]
    logger.info(f"\nTrack length stats:")
    logger.info(f"  Mean: {np.mean(track_lens):.2f}")
    logger.info(f"  Median: {np.median(track_lens):.1f}")
    logger.info(f"  Max: {np.max(track_lens)}")

    # Reprojection error distribution
    errors = [p.error for p in sfm_map.points3D.values()]
    logger.info(f"\nReprojection error stats:")
    logger.info(f"  Mean: {np.mean(errors):.4f} px")
    logger.info(f"  Median: {np.median(errors):.4f} px")
    logger.info(f"  95th: {np.percentile(errors, 95):.4f} px")

    # Per-image observation count
    obs_counts = [np.sum(img.point3D_ids >= 0) for img in sfm_map.images.values()]
    logger.info(f"\nPer-image 3D observations:")
    logger.info(f"  Mean: {np.mean(obs_counts):.1f}")
    logger.info(f"  Min: {np.min(obs_counts)}")
    logger.info(f"  Max: {np.max(obs_counts)}")

    return {
        "num_cameras": len(sfm_map.cameras),
        "num_images": len(sfm_map.images),
        "num_points3D": len(sfm_map.points3D),
        "mean_track_length": float(np.mean(track_lens)),
        "mean_reproj_error": float(np.mean(errors)),
        "camera_extent_xyz": [
            float(positions[:,i].max() - positions[:,i].min()) for i in range(3)
        ],
    }


def test_self_localization(sfm_map: SfMMap, db_image_dir: str, num_test: int = 5):
    """Self-localization 테스트: DB 이미지로 자기 자신을 PnP 추정"""
    logger.info("\n" + "=" * 60)
    logger.info("Self-Localization Test (PnP validation)")
    logger.info("=" * 60)

    localizer = PnPLocalizer(sfm_map)
    db_dir = Path(db_image_dir)

    results = []
    test_images = list(sfm_map.images.values())[:num_test]

    for db_img in test_images:
        query_path = db_dir / db_img.name
        if not query_path.exists():
            logger.warning(f"Image not found: {query_path}")
            continue

        logger.info(f"\nTesting: {db_img.name}")
        logger.info(f"  GT position: {db_img.position}")

        result = localizer.localize_with_db_images(
            str(query_path), str(db_dir),
            top_k=5, ransac_threshold=8.0, min_inliers=10
        )

        if result:
            est_pos = np.array(result["position_xyz"])
            gt_pos = db_img.position
            pos_error = np.linalg.norm(est_pos - gt_pos)
            logger.info(f"  Est position: {est_pos}")
            logger.info(f"  Position error: {pos_error:.4f} (SfM units)")
            logger.info(f"  Inliers: {result['num_inliers']}")
            logger.info(f"  Reproj error: {result['mean_reproj_error_px']:.2f} px")
            result["gt_position"] = gt_pos.tolist()
            result["position_error"] = float(pos_error)
            results.append({"image": db_img.name, **result})
        else:
            logger.warning(f"  FAILED to localize")
            results.append({"image": db_img.name, "status": "failed"})

    return results


def extract_all_sfm_poses(sfm_map: SfMMap) -> dict:
    """SfM에서 모든 등록된 이미지의 pose를 추출"""
    logger.info("\n" + "=" * 60)
    logger.info("Extracting all SfM poses")
    logger.info("=" * 60)

    poses = {}
    for img_id, img in sfm_map.images.items():
        T_c2w = img.T_cam_to_world
        from scripts.phase1_sfm.sfm_map import _rotmat2qvec
        qvec = _rotmat2qvec(T_c2w[:3, :3])

        poses[img.name] = {
            "image_id": img.image_id,
            "T_cam_to_world": T_c2w.tolist(),
            "quaternion_wxyz": qvec.tolist(),
            "position_xyz": img.position.tolist(),
            "num_observations": int(np.sum(img.point3D_ids >= 0)),
        }
        logger.info(f"  {img.name}: pos={img.position}, obs={np.sum(img.point3D_ids >= 0)}")

    return poses


def main():
    parser = argparse.ArgumentParser(description="Phase 1: SfM validation & PnP testing")
    parser.add_argument("--sfm_model_path", required=True, help="COLMAP TXT model dir")
    parser.add_argument("--db_image_dir", required=True, help="Original images dir")
    parser.add_argument("--output_dir", default="temp_data/phase1_results", help="Output dir")
    parser.add_argument("--test_count", type=int, default=5, help="Number of self-loc tests")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SfM map
    logger.info("Loading SfM map...")
    sfm_map = SfMMap(args.sfm_model_path)

    # 2. Validate
    val_stats = validate_sfm_map(sfm_map)
    with open(output_dir / "sfm_validation.json", 'w') as f:
        json.dump(val_stats, f, indent=2)

    # 3. Extract all SfM poses
    all_poses = extract_all_sfm_poses(sfm_map)
    save_poses_json(all_poses, str(output_dir / "sfm_all_poses.json"))

    # 4. Self-localization test
    test_results = test_self_localization(sfm_map, args.db_image_dir, args.test_count)
    with open(output_dir / "self_localization_test.json", 'w') as f:
        json.dump(test_results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
