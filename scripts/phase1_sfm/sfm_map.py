"""
SfM Map Loader - COLMAP TXT format parser
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        if self.model in ("OPENCV", "PINHOLE"):
            fx, fy, cx, cy = self.params[:4]
        elif self.model == "SIMPLE_RADIAL":
            f, cx, cy = self.params[:3]
            fx = fy = f
        else:
            fx, fy, cx, cy = self.params[:4]
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

    @property
    def dist_coeffs(self) -> np.ndarray:
        if self.model == "OPENCV":
            return self.params[4:8].astype(np.float64)
        elif self.model == "SIMPLE_RADIAL":
            k = self.params[3] if len(self.params) > 3 else 0.0
            return np.array([k, 0, 0, 0], dtype=np.float64)
        return np.zeros(4, dtype=np.float64)


@dataclass
class Image:
    image_id: int
    qw: float; qx: float; qy: float; qz: float
    tx: float; ty: float; tz: float
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray

    @property
    def R(self) -> np.ndarray:
        return _qvec2rotmat(np.array([self.qw, self.qx, self.qy, self.qz]))

    @property
    def t(self) -> np.ndarray:
        return np.array([self.tx, self.ty, self.tz], dtype=np.float64)

    @property
    def T_cam_to_world(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return np.linalg.inv(T)

    @property
    def position(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class Point3D:
    point3D_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: List[Tuple[int, int]]


class SfMMap:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.cameras: Dict[int, Camera] = {}
        self.images: Dict[int, Image] = {}
        self.points3D: Dict[int, Point3D] = {}
        self._load()

    def _load(self):
        self._load_cameras()
        self._load_images()
        self._load_points3D()
        logger.info(f"Loaded: {len(self.cameras)} cam, {len(self.images)} img, {len(self.points3D)} pts")

    def _load_cameras(self):
        with open(self.model_path / "cameras.txt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split()
                self.cameras[int(parts[0])] = Camera(
                    int(parts[0]), parts[1], int(parts[2]), int(parts[3]),
                    np.array([float(p) for p in parts[4:]]))

    def _load_images(self):
        with open(self.model_path / "images.txt") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            obs = lines[i+1].split()
            n = len(obs) // 3
            xys = np.zeros((n, 2), dtype=np.float64)
            pids = np.full(n, -1, dtype=np.int64)
            for j in range(n):
                xys[j] = [float(obs[3*j]), float(obs[3*j+1])]
                pids[j] = int(float(obs[3*j+2]))
            self.images[int(parts[0])] = Image(
                int(parts[0]), *map(float, parts[1:5]), *map(float, parts[5:8]),
                int(parts[8]), parts[9], xys, pids)

    def _load_points3D(self):
        with open(self.model_path / "points3D.txt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split()
                pid = int(parts[0])
                track = [(int(parts[k]), int(parts[k+1])) for k in range(8, len(parts), 2)]
                self.points3D[pid] = Point3D(
                    pid, np.array([float(parts[1]),float(parts[2]),float(parts[3])]),
                    np.array([int(parts[4]),int(parts[5]),int(parts[6])]),
                    float(parts[7]), track)

    def get_3d_2d_correspondences(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        img = self.images[image_id]
        mask = img.point3D_ids >= 0
        pts3d, pts2d = [], []
        for pid, xy in zip(img.point3D_ids[mask], img.xys[mask]):
            if pid in self.points3D:
                pts3d.append(self.points3D[pid].xyz)
                pts2d.append(xy)
        return np.array(pts3d, dtype=np.float64), np.array(pts2d, dtype=np.float64)

    def get_all_3d_points(self) -> np.ndarray:
        return np.array([p.xyz for p in self.points3D.values()], dtype=np.float64)

    def get_image_by_name(self, name: str) -> Optional[Image]:
        for img in self.images.values():
            if img.name == name: return img
        return None


def _qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]], dtype=np.float64)


def _rotmat2qvec(R):
    tr = R.trace()
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])
