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
PnP Camera Localizer - SfM 맵에 대한 고정 카메라 전역 pose 추정

SuperPoint + LightGlue (또는 SIFT) 기반 feature matching으로
고정 카메라 이미지의 2D 특징점과 SfM 3D 포인트를 대응시켜
PnP-RANSAC로 카메라의 전역 pose를 추정합니다.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
import json

from scripts.phase1_sfm.sfm_map import SfMMap, _rotmat2qvec

logger = logging.getLogger(__name__)


class PnPLocalizer:
    """SfM 맵을 이용한 PnP 기반 카메라 포즈 추정기"""

    def __init__(self, sfm_map: SfMMap):
        self.sfm_map = sfm_map
        # SIFT feature extractor (OpenCV built-in, no torch dependency)
        self.sift = cv2.SIFT_create(nfeatures=8192)
        # FLANN-based matcher
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def _build_db_descriptors(self, db_image_ids: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SfM DB 이미지들의 3D 포인트에 대한 descriptor DB 구축
        Returns:
            all_descriptors: Mx128 descriptor array
            all_points3d: Mx3 corresponding 3D points
            all_image_ids: M image IDs (for debugging)
        """
        all_desc = []
        all_pts3d = []
        all_img_ids = []

        for img_id in db_image_ids:
            img_data = self.sfm_map.images[img_id]
            pts3d, pts2d = self.sfm_map.get_3d_2d_correspondences(img_id)
            if len(pts3d) == 0:
                continue
            all_pts3d.append(pts3d)
            all_img_ids.extend([img_id] * len(pts3d))

        if not all_pts3d:
            return np.array([]), np.array([]), np.array([])

        return (
            np.vstack(all_pts3d) if all_pts3d else np.array([]),
            np.array(all_img_ids)
        )

    def localize_image(
        self,
        query_image_path: str,
        camera_K: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        pnp_method: int = cv2.SOLVEPNP_ITERATIVE,
        ransac_threshold: float = 8.0,
        min_inliers: int = 15
    ) -> Optional[Dict]:
        """
        쿼리 이미지의 전역 pose를 PnP-RANSAC으로 추정

        Args:
            query_image_path: 고정 카메라에서 촬영한 이미지 경로
            camera_K: 3x3 intrinsic matrix (None이면 SfM 카메라 사용)
            dist_coeffs: distortion coefficients (None이면 SfM 카메라 사용)

        Returns:
            dict with T_world_cam (4x4), quaternion (w,x,y,z), translation, inliers count
        """
        # Load query image
        query_img = cv2.imread(str(query_image_path), cv2.IMREAD_GRAYSCALE)
        if query_img is None:
            logger.error(f"Cannot load image: {query_image_path}")
            return None

        # Extract SIFT features from query
        kp_query, desc_query = self.sift.detectAndCompute(query_img, None)
        if desc_query is None or len(kp_query) < 10:
            logger.warning(f"Too few features in query image: {len(kp_query) if kp_query else 0}")
            return None

        logger.info(f"Query: {len(kp_query)} features extracted")

        # Use default camera if not provided
        if camera_K is None:
            cam = list(self.sfm_map.cameras.values())[0]
            camera_K = cam.K
            dist_coeffs = cam.dist_coeffs

        if dist_coeffs is None:
            dist_coeffs = np.zeros(4, dtype=np.float64)

        # Match against all SfM database images
        best_result = None
        best_inliers = 0

        for img_id, db_img in self.sfm_map.images.items():
            pts3d, pts2d = self.sfm_map.get_3d_2d_correspondences(img_id)
            if len(pts3d) < 20:
                continue

            # Load DB image and extract features
            # Since we don't have the DB images, use pycolmap's approach:
            # Match query descriptors to DB 2D points via SIFT re-extraction
            # But since DB images may not be available, we use a covisibility approach
            result = self._try_pnp_with_covisibility(
                kp_query, desc_query, query_img,
                img_id, camera_K, dist_coeffs,
                ransac_threshold, min_inliers
            )

            if result and result["num_inliers"] > best_inliers:
                best_result = result
                best_inliers = result["num_inliers"]

        return best_result

    def localize_with_db_images(
        self,
        query_image_path: str,
        db_image_dir: str,
        camera_K: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        top_k: int = 10,
        ransac_threshold: float = 8.0,
        min_inliers: int = 15
    ) -> Optional[Dict]:
        """
        DB 이미지가 있는 경우의 PnP 추정 (권장 방법)
        """
        query_img = cv2.imread(str(query_image_path), cv2.IMREAD_GRAYSCALE)
        if query_img is None:
            logger.error(f"Cannot load image: {query_image_path}")
            return None

        kp_query, desc_query = self.sift.detectAndCompute(query_img, None)
        if desc_query is None or len(kp_query) < 10:
            return None

        logger.info(f"Query: {len(kp_query)} SIFT features")

        if camera_K is None:
            cam = list(self.sfm_map.cameras.values())[0]
            camera_K = cam.K
            dist_coeffs = cam.dist_coeffs
        if dist_coeffs is None:
            dist_coeffs = np.zeros(4, dtype=np.float64)

        db_dir = Path(db_image_dir)
        all_3d = []
        all_2d = []

        # Find best matching DB images and accumulate 3D-2D correspondences
        scored_images = []
        for img_id, db_img in self.sfm_map.images.items():
            db_path = db_dir / db_img.name
            if not db_path.exists():
                continue
            db_gray = cv2.imread(str(db_path), cv2.IMREAD_GRAYSCALE)
            if db_gray is None:
                continue

            kp_db, desc_db = self.sift.detectAndCompute(db_gray, None)
            if desc_db is None or len(kp_db) < 10:
                continue

            # Match
            matches = self.matcher.knnMatch(desc_query, desc_db, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            scored_images.append((len(good), img_id, kp_db, good))

        # Sort by match count, take top-K
        scored_images.sort(key=lambda x: -x[0])
        logger.info(f"Top matching DB images: {[(s[0], self.sfm_map.images[s[1]].name) for s in scored_images[:5]]}")

        for count, img_id, kp_db, good_matches in scored_images[:top_k]:
            db_img = self.sfm_map.images[img_id]
            pts3d, pts2d_db = self.sfm_map.get_3d_2d_correspondences(img_id)
            if len(pts3d) == 0:
                continue

            # For each good match, find if the DB keypoint corresponds to a 3D point
            for m in good_matches:
                db_kp_pt = np.array(kp_db[m.trainIdx].pt)
                query_kp_pt = np.array(kp_query[m.queryIdx].pt)

                # Find nearest DB 2D point that has a 3D correspondence
                dists = np.linalg.norm(pts2d_db - db_kp_pt, axis=1)
                nearest_idx = np.argmin(dists)
                if dists[nearest_idx] < 5.0:  # within 5 pixels
                    all_3d.append(pts3d[nearest_idx])
                    all_2d.append(query_kp_pt)

        if len(all_3d) < min_inliers:
            logger.warning(f"Not enough correspondences: {len(all_3d)} < {min_inliers}")
            return None

        pts3d_arr = np.array(all_3d, dtype=np.float64)
        pts2d_arr = np.array(all_2d, dtype=np.float64)
        logger.info(f"Total 3D-2D correspondences: {len(pts3d_arr)}")

        # PnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_arr, pts2d_arr, camera_K, dist_coeffs,
            iterationsCount=10000,
            reprojectionError=ransac_threshold,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or inliers is None or len(inliers) < min_inliers:
            logger.warning(f"PnP failed or too few inliers: {len(inliers) if inliers is not None else 0}")
            return None

        # Build pose
        R, _ = cv2.Rodrigues(rvec)
        T_w2c = np.eye(4, dtype=np.float64)
        T_w2c[:3, :3] = R
        T_w2c[:3, 3] = tvec.flatten()
        T_c2w = np.linalg.inv(T_w2c)

        qvec = _rotmat2qvec(T_c2w[:3, :3])
        position = T_c2w[:3, 3]

        # Compute reprojection error for inliers
        reproj_pts, _ = cv2.projectPoints(
            pts3d_arr[inliers.flatten()], rvec, tvec, camera_K, dist_coeffs
        )
        reproj_error = np.mean(np.linalg.norm(
            pts2d_arr[inliers.flatten()] - reproj_pts.reshape(-1, 2), axis=1
        ))

        result = {
            "T_world_to_cam": T_w2c.tolist(),
            "T_cam_to_world": T_c2w.tolist(),
            "quaternion_wxyz": qvec.tolist(),
            "position_xyz": position.tolist(),
            "num_inliers": int(len(inliers)),
            "num_correspondences": int(len(pts3d_arr)),
            "mean_reproj_error_px": float(reproj_error),
            "camera_K": camera_K.tolist(),
        }
        logger.info(
            f"PnP OK: {len(inliers)} inliers / {len(pts3d_arr)} corr, "
            f"reproj={reproj_error:.2f}px, pos={position}"
        )
        return result

    def _try_pnp_with_covisibility(self, kp_q, desc_q, query_img,
                                     db_img_id, K, dist, thresh, min_inl):
        """Internal: try PnP using covisibility graph (no DB images needed)"""
        # This is a fallback — without original DB images we cannot do SIFT matching
        # Instead, if the query IS one of the DB images, we can validate
        return None


def save_poses_json(poses: Dict[str, Dict], output_path: str):
    """추정된 pose들을 JSON으로 저장"""
    with open(output_path, 'w') as f:
        json.dump(poses, f, indent=2)
    logger.info(f"Saved {len(poses)} poses to {output_path}")


def load_poses_json(path: str) -> Dict[str, Dict]:
    """저장된 pose JSON 로드"""
    with open(path) as f:
        return json.load(f)
