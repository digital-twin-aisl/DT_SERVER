"""
HLoc-style PnP Localizer — SuperPoint + LightGlue 기반 고정 카메라 전역 Pose 추정

SIFT 대신 SuperPoint (learned feature) + LightGlue (learned matcher)를 사용하여
고정 카메라 이미지를 SfM 3D 맵에 대해 localize합니다.

성능 비교 (일반적인 벤치마크):
  - SIFT + ratio test:      ~70% recall @5cm
  - SuperPoint + LightGlue: ~90% recall @5cm (특히 viewpoint/illumination 변화에 강건)
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
import json

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

from scripts.phase1_sfm.sfm_map import SfMMap, _rotmat2qvec

logger = logging.getLogger(__name__)


class HLocLocalizer:
    """SuperPoint + LightGlue 기반 PnP 카메라 Localizer"""

    def __init__(
        self,
        sfm_map: SfMMap,
        device: str = 'cuda',
        max_keypoints: int = 2048,
    ):
        self.sfm_map = sfm_map
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # ── SuperPoint feature extractor ──────────────────
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)

        # ── LightGlue matcher ────────────────────────────
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        logger.info(
            f"HLocLocalizer initialized: device={self.device}, "
            f"max_keypoints={max_keypoints}"
        )

        # ── DB feature cache (lazy-loaded) ────────────────
        self._db_features_cache: Dict[int, dict] = {}

    @torch.no_grad()
    def _extract_features(self, image_path: str) -> dict:
        """이미지에서 SuperPoint feature 추출"""
        image = load_image(image_path).to(self.device)
        feats = self.extractor.extract(image)
        return feats

    @torch.no_grad()
    def _match_features(self, feats0: dict, feats1: dict) -> dict:
        """두 feature set 간 LightGlue 매칭"""
        return self.matcher({'image0': feats0, 'image1': feats1})

    def build_db_features(self, db_image_dir: str, max_images: int = -1):
        """
        SfM DB 이미지들의 SuperPoint feature를 미리 추출하여 캐싱

        Args:
            db_image_dir: 원본 이미지 디렉토리
            max_images: 최대 처리 이미지 수 (-1이면 전체)
        """
        db_dir = Path(db_image_dir)
        count = 0
        for img_id, img_data in self.sfm_map.images.items():
            if 0 < max_images <= count:
                break
            img_path = db_dir / img_data.name
            if not img_path.exists():
                continue

            feats = self._extract_features(str(img_path))
            self._db_features_cache[img_id] = feats
            count += 1

        logger.info(f"Built DB features cache: {count} images")

    def localize(
        self,
        query_image_path: str,
        db_image_dir: str,
        camera_K: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        top_k: int = 10,
        ransac_threshold: float = 12.0,
        min_inliers: int = 15,
    ) -> Optional[Dict]:
        """
        쿼리 이미지의 전역 pose를 SuperPoint+LightGlue+PnP로 추정

        Args:
            query_image_path: 고정 카메라 촬영 이미지
            db_image_dir: SfM DB 이미지 디렉토리
            camera_K: 3x3 intrinsic (None → SfM 카메라 사용)
            dist_coeffs: 왜곡 계수 (None → SfM 카메라 사용)
            top_k: 매칭할 상위 DB 이미지 수
            ransac_threshold: PnP RANSAC 임계값 (px)
            min_inliers: 최소 inlier 수

        Returns:
            dict with pose, inliers, error metrics or None
        """
        # Defaults
        if camera_K is None:
            cam = list(self.sfm_map.cameras.values())[0]
            camera_K = cam.K
            dist_coeffs = cam.dist_coeffs
        if dist_coeffs is None:
            dist_coeffs = np.zeros(4, dtype=np.float64)

        # Extract query features
        query_feats = self._extract_features(query_image_path)
        query_kps = query_feats['keypoints'][0].cpu().numpy()  # Nx2
        logger.info(f"Query: {len(query_kps)} SuperPoint features from {Path(query_image_path).name}")

        db_dir = Path(db_image_dir)

        # Build DB features if not cached
        if not self._db_features_cache:
            self.build_db_features(db_image_dir)

        # ── Score DB images by match count ──────────────
        scored: List[Tuple[int, int, np.ndarray]] = []  # (match_count, img_id, matches01)

        for img_id, db_feats in self._db_features_cache.items():
            match_result = self._match_features(query_feats, db_feats)
            matches01 = match_result['matches'][0].cpu().numpy()  # Mx2
            valid = matches01[:, 0] >= 0
            n_matches = valid.sum()
            if n_matches > 0:
                scored.append((int(n_matches), img_id, matches01[valid]))

        scored.sort(key=lambda x: -x[0])
        if scored:
            logger.info(
                f"Top DB matches: "
                f"{[(s[0], self.sfm_map.images[s[1]].name) for s in scored[:5]]}"
            )

        # ── Accumulate 3D-2D correspondences from top-K ──
        all_3d, all_2d = [], []

        for count, img_id, matches01 in scored[:top_k]:
            db_img = self.sfm_map.images[img_id]
            pts3d_db, pts2d_db = self.sfm_map.get_3d_2d_correspondences(img_id)
            if len(pts3d_db) == 0:
                continue

            db_feats = self._db_features_cache[img_id]
            db_kps = db_feats['keypoints'][0].cpu().numpy()

            for q_idx, d_idx in matches01:
                query_pt = query_kps[q_idx]
                db_pt = db_kps[d_idx]

                # Find nearest SfM 2D point to the DB keypoint
                dists = np.linalg.norm(pts2d_db - db_pt, axis=1)
                nearest = np.argmin(dists)
                if dists[nearest] < 5.0:  # within 5px
                    all_3d.append(pts3d_db[nearest])
                    all_2d.append(query_pt)

        if len(all_3d) < min_inliers:
            logger.warning(f"Not enough correspondences: {len(all_3d)} < {min_inliers}")
            return None

        pts3d = np.array(all_3d, dtype=np.float64)
        pts2d = np.array(all_2d, dtype=np.float64)
        logger.info(f"Total 3D-2D correspondences: {len(pts3d)}")

        # ── PnP-RANSAC ───────────────────────────────────
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, camera_K, dist_coeffs,
            iterationsCount=10000,
            reprojectionError=ransac_threshold,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < min_inliers:
            logger.warning(f"PnP failed: {len(inliers) if inliers is not None else 0} inliers")
            return None

        # ── Build result ─────────────────────────────────
        R, _ = cv2.Rodrigues(rvec)
        T_w2c = np.eye(4, dtype=np.float64)
        T_w2c[:3, :3] = R
        T_w2c[:3, 3] = tvec.flatten()
        T_c2w = np.linalg.inv(T_w2c)

        qvec = _rotmat2qvec(T_c2w[:3, :3])
        position = T_c2w[:3, 3]

        reproj_pts, _ = cv2.projectPoints(
            pts3d[inliers.flatten()], rvec, tvec, camera_K, dist_coeffs
        )
        reproj_error = float(np.mean(np.linalg.norm(
            pts2d[inliers.flatten()] - reproj_pts.reshape(-1, 2), axis=1
        )))

        result = {
            "method": "SuperPoint+LightGlue+PnP",
            "T_world_to_cam": T_w2c.tolist(),
            "T_cam_to_world": T_c2w.tolist(),
            "quaternion_wxyz": qvec.tolist(),
            "position_xyz": position.tolist(),
            "num_inliers": int(len(inliers)),
            "num_correspondences": int(len(pts3d)),
            "mean_reproj_error_px": reproj_error,
        }

        logger.info(
            f"PnP OK: {len(inliers)} inliers / {len(pts3d)} corr, "
            f"reproj={reproj_error:.2f}px, pos={position}"
        )
        return result


def run_hloc_localization(
    sfm_model_path: str,
    db_image_dir: str,
    query_image_path: str,
    output_path: Optional[str] = None,
) -> Optional[Dict]:
    """Convenience function: SfM 맵 로드 → HLoc localize → 결과 반환"""
    sfm_map = SfMMap(sfm_model_path)
    localizer = HLocLocalizer(sfm_map)
    result = localizer.localize(query_image_path, db_image_dir)
    if result and output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    return result
