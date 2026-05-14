#!/usr/bin/env python3
"""
Phase 1 HLoc 통합 테스트 — SuperPoint+LightGlue PnP vs SIFT PnP 비교

사용법:
  MKL_THREADING_LAYER=GNU python -m scripts.phase1_sfm.test_hloc \
    --sfm_model_path temp_data/sfm_output/sparse_txt \
    --db_image_dir Collabolab \
    --test_count 5
"""
import sys
import json
import argparse
import logging
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1_sfm.sfm_map import SfMMap
from scripts.phase1_sfm.hloc_localizer import HLocLocalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_model_path", required=True)
    parser.add_argument("--db_image_dir", required=True)
    parser.add_argument("--test_count", type=int, default=5)
    parser.add_argument("--output_dir", default="temp_data/phase1_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SfM map
    sfm_map = SfMMap(args.sfm_model_path)
    logger.info(f"SfM map: {len(sfm_map.images)} images, {len(sfm_map.points3D)} pts")

    # Init HLoc localizer
    localizer = HLocLocalizer(sfm_map, max_keypoints=2048)

    # Pre-build DB features (one-time cost)
    t0 = time.time()
    localizer.build_db_features(args.db_image_dir)
    db_build_time = time.time() - t0
    logger.info(f"DB feature extraction: {db_build_time:.1f}s")

    # Run self-localization tests
    db_dir = Path(args.db_image_dir)
    test_images = list(sfm_map.images.values())[:args.test_count]
    results = []

    logger.info(f"\n{'='*60}")
    logger.info(f"HLoc Self-Localization Test ({len(test_images)} images)")
    logger.info(f"{'='*60}")

    for img in test_images:
        query_path = db_dir / img.name
        if not query_path.exists():
            continue

        logger.info(f"\nTesting: {img.name}")
        gt_pos = img.position

        t0 = time.time()
        result = localizer.localize(
            str(query_path), str(db_dir),
            top_k=5, ransac_threshold=12.0, min_inliers=10,
        )
        elapsed = time.time() - t0

        if result:
            est_pos = np.array(result["position_xyz"])
            pos_error = float(np.linalg.norm(est_pos - gt_pos))
            result["gt_position"] = gt_pos.tolist()
            result["position_error"] = pos_error
            result["localization_time_sec"] = elapsed
            results.append({"image": img.name, **result})

            logger.info(f"  ✓ Position error: {pos_error:.4f}")
            logger.info(f"  Inliers: {result['num_inliers']}, Reproj: {result['mean_reproj_error_px']:.2f}px")
            logger.info(f"  Time: {elapsed:.2f}s")
        else:
            results.append({"image": img.name, "status": "failed"})
            logger.warning(f"  ✗ FAILED")

    # Summary
    successful = [r for r in results if "position_error" in r]
    if successful:
        errors = [r["position_error"] for r in successful]
        times = [r["localization_time_sec"] for r in successful]
        logger.info(f"\n{'='*60}")
        logger.info(f"Summary: {len(successful)}/{len(results)} succeeded")
        logger.info(f"  Mean pos error: {np.mean(errors):.4f}")
        logger.info(f"  Max pos error:  {np.max(errors):.4f}")
        logger.info(f"  Mean time:      {np.mean(times):.2f}s")
        logger.info(f"{'='*60}")

    # Save
    out_path = output_dir / "hloc_localization_test.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
