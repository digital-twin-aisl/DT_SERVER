"""Walkable height queries for Z-up USD ground meshes."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


GROUND_CACHE_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class GroundSurface:
    """Triangulated USD surface stored in VoxelPose millimetres."""

    triangles_mm: np.ndarray
    _torch_cache: dict[tuple[Any, Any], Any] = field(default_factory=dict)

    @classmethod
    def from_cache(
        cls,
        cache_path: str | Path,
        source_usd_path: str | Path | None = None,
    ) -> "GroundSurface":
        cache_path = Path(cache_path)
        with np.load(cache_path, allow_pickle=False) as cache:
            version = int(cache["format_version"].item())
            if version != GROUND_CACHE_VERSION:
                raise ValueError(
                    f"unsupported Ground cache version {version}: {cache_path}"
                )
            triangles = np.asarray(cache["triangles_mm"], dtype=np.float64)
            source_sha256 = str(cache["source_sha256"].item())

        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError(
                f"Ground cache triangles must have shape [N, 3, 3]: {cache_path}"
            )
        if len(triangles) == 0 or not np.isfinite(triangles).all():
            raise ValueError(f"Ground cache contains invalid triangles: {cache_path}")

        if source_usd_path is not None:
            source_path = Path(source_usd_path)
            actual_sha256 = _sha256(source_path)
            if source_sha256 != actual_sha256:
                raise ValueError(
                    "Ground cache does not match its source USD; regenerate "
                    f"{cache_path} from {source_path}"
                )
        return cls(triangles)

    @classmethod
    def from_usd(cls, path: str | Path) -> "GroundSurface":
        try:
            from pxr import Gf, Usd, UsdGeom
        except ImportError as exc:
            raise RuntimeError(
                "OpenUSD Python bindings are required to read the Ground USD"
            ) from exc

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise ValueError(f"failed to open Ground USD: {path}")
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise ValueError("Ground USD must use a Z-up coordinate system")
        millimetres_per_stage_unit = UsdGeom.GetStageMetersPerUnit(stage) * 1000.0
        xform_cache = UsdGeom.XformCache()
        triangles = []

        for prim in stage.Traverse():
            if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            transform = xform_cache.GetLocalToWorldTransform(prim)
            points = np.asarray(
                [
                    tuple(transform.Transform(Gf.Vec3d(*point)))
                    for point in (mesh.GetPointsAttr().Get() or [])
                ],
                dtype=np.float64,
            )
            points *= millimetres_per_stage_unit
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            indices = mesh.GetFaceVertexIndicesAttr().Get() or []
            offset = 0
            for count in counts:
                face = indices[offset : offset + count]
                offset += count
                for index in range(1, count - 1):
                    triangles.append(
                        points[[face[0], face[index], face[index + 1]]]
                    )

        if not triangles:
            raise ValueError(f"Ground USD contains no mesh triangles: {path}")
        return cls(np.asarray(triangles, dtype=np.float64))

    def save_cache(
        self,
        cache_path: str | Path,
        source_usd_path: str | Path,
    ) -> None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            format_version=np.asarray(GROUND_CACHE_VERSION, dtype=np.int64),
            source_sha256=np.asarray(_sha256(Path(source_usd_path))),
            triangles_mm=self.triangles_mm,
        )

    def height_mm(self, x_mm: float, y_mm: float) -> float:
        values = self.heights_mm(np.asarray([[x_mm, y_mm]], dtype=np.float64))
        return float(values[0])

    def heights_mm(self, xy_mm: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy_mm, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy_mm must have shape [N, 2]")

        triangles = self.triangles_mm
        a = triangles[:, 0]
        b = triangles[:, 1]
        c = triangles[:, 2]
        denominator = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (
            c[:, 0] - b[:, 0]
        ) * (a[:, 1] - c[:, 1])
        usable = np.abs(denominator) > 1e-9
        safe_denominator = np.where(usable, denominator, 1.0)

        x = xy[:, 0:1]
        y = xy[:, 1:2]
        u = (
            (b[:, 1] - c[:, 1]) * (x - c[:, 0])
            + (c[:, 0] - b[:, 0]) * (y - c[:, 1])
        ) / safe_denominator
        v = (
            (c[:, 1] - a[:, 1]) * (x - c[:, 0])
            + (a[:, 0] - c[:, 0]) * (y - c[:, 1])
        ) / safe_denominator
        inside = usable & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1.0 + 1e-7)
        z = u * a[:, 2] + v * b[:, 2] + (1.0 - u - v) * c[:, 2]
        z = np.where(inside, z, -np.inf)
        heights = z.max(axis=1)
        heights[~np.isfinite(heights)] = np.nan
        return heights

    def heights_mm_torch(self, xy_mm):
        """Torch equivalent used for a small number of predicted foot joints."""
        import torch

        if xy_mm.ndim != 2 or xy_mm.shape[1] != 2:
            raise ValueError("xy_mm must have shape [N, 2]")
        key = (xy_mm.device, xy_mm.dtype)
        triangles = self._torch_cache.get(key)
        if triangles is None:
            triangles = torch.as_tensor(
                self.triangles_mm,
                device=xy_mm.device,
                dtype=xy_mm.dtype,
            )
            self._torch_cache[key] = triangles

        a = triangles[:, 0]
        b = triangles[:, 1]
        c = triangles[:, 2]
        denominator = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (
            c[:, 0] - b[:, 0]
        ) * (a[:, 1] - c[:, 1])
        usable = denominator.abs() > 1e-6
        safe_denominator = torch.where(usable, denominator, torch.ones_like(denominator))
        x = xy_mm[:, 0:1]
        y = xy_mm[:, 1:2]
        u = (
            (b[:, 1] - c[:, 1]) * (x - c[:, 0])
            + (c[:, 0] - b[:, 0]) * (y - c[:, 1])
        ) / safe_denominator
        v = (
            (c[:, 1] - a[:, 1]) * (x - c[:, 0])
            + (a[:, 0] - c[:, 0]) * (y - c[:, 1])
        ) / safe_denominator
        inside = usable & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
        z = u * a[:, 2] + v * b[:, 2] + (1.0 - u - v) * c[:, 2]
        z = torch.where(inside, z, torch.full_like(z, -torch.inf))
        heights = z.max(dim=1).values
        return torch.where(
            torch.isfinite(heights),
            heights,
            torch.full_like(heights, torch.nan),
        )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Ground USD cache utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("source_usd", type=Path)
    export_parser.add_argument("output_cache", type=Path)
    args = parser.parse_args()

    if args.command == "export":
        surface = GroundSurface.from_usd(args.source_usd)
        surface.save_cache(args.output_cache, args.source_usd)
        print(f"wrote {len(surface.triangles_mm)} triangles to {args.output_cache}")


if __name__ == "__main__":
    _main()
