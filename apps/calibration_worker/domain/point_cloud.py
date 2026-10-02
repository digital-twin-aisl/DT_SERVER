# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Bounded VGGT depth cloud export in the same metric frame as final cameras."""
import hashlib
import math
from pathlib import Path
import shutil

import numpy as np

WORLD_FRAME = 'marker-tree USD Z-up metres'


def export_depth_cloud(path, predictions, alignment, sources, *, min_confidence=1.,
                       max_points=1_000_000, ba_applied=False):
    depth = np.asarray(predictions['depth'])
    confidence = np.asarray(predictions['depth_conf'])
    images = np.asarray(predictions['images'])
    if depth.ndim != 3 or confidence.shape != depth.shape or images.shape != (*depth.shape, 3):
        raise ValueError('Cloud depth/confidence/RGB dimensions do not match.')
    n, height, width = depth.shape
    regions = np.asarray(predictions.get('valid_regions', np.tile([0,0,width,height],(n,1))))
    if (regions.shape != (n,4) or not np.isfinite(regions).all()
            or np.any(regions[:,:2] < 0) or np.any(regions[:,2:] > [width,height])
            or np.any(regions[:,2:] <= regions[:,:2])):
        raise ValueError('Invalid unpadded image regions.')
    if len(sources) != n or not 1 <= max_points <= 2_000_000 or not math.isfinite(min_confidence):
        raise ValueError('Invalid cloud sources, budget or confidence threshold.')
    # Allocate a deterministic, spatially spread budget to every input image.
    budget = max_points // n
    if not budget: raise ValueError('Cloud budget must cover every source image.')
    stride = max(1, math.ceil(math.sqrt(height * width / budget)))
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    xy = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)]
    positions, colors, scores, indices = [], [], [], []
    for i in range(n):
        d, conf = depth[i, yy, xx].ravel(), confidence[i, yy, xx].ravel()
        rgb = images[i, yy, xx].reshape(-1, 3)
        valid = np.isfinite(d) & (d > 0) & np.isfinite(conf) & (conf >= min_confidence) & np.isfinite(rgb).all(axis=1)
        left, top, right, bottom = regions[i]
        valid &= (xy[:,0]>=left)&(xy[:,0]<right)&(xy[:,1]>=top)&(xy[:,1]<bottom)
        selected = np.flatnonzero(valid)
        if len(selected) > budget:
            selected = selected[np.linspace(0, len(selected)-1, budget, dtype=int)]
        # Depth is coupled to VGGT's predicted K, not the calibrated BA K.
        rays = xy[selected] @ np.linalg.inv(predictions['intrinsics'][i]).T
        camera_xyz = rays * d[selected, None]
        extrinsic = np.asarray(predictions['extrinsics'][i], dtype=np.float64)
        reconstruction_xyz = (camera_xyz - extrinsic[:3, 3]) @ extrinsic[:3, :3]
        world = alignment.transform_points(reconstruction_xyz)
        finite = np.isfinite(world).all(axis=1) & (np.abs(world) < 1e6).all(axis=1)
        positions.append(world[finite].astype(np.float32))
        colors.append(np.rint(np.clip(rgb[selected][finite], 0, 1)*255).astype(np.uint8))
        scores.append(conf[selected][finite].astype(np.float32))
        indices.append(np.full(finite.sum(), i, dtype=np.uint32))
    xyz = np.concatenate(positions)
    if not len(xyz): raise ValueError('No valid points survived cloud filtering.')
    path = Path(path)
    np.savez_compressed(path, xyz=xyz, rgb=np.concatenate(colors),
                        confidence=np.concatenate(scores), source_index=np.concatenate(indices),
                        reconstruction_to_world=alignment.matrix4(),
                        depth_intrinsics=predictions['intrinsics'],
                        final_extrinsics=predictions['extrinsics'])
    return dict(schema_version=1, file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                coordinate_system=WORLD_FRAME, point_count=len(xyz), sources=sources,
                min_confidence=float(min_confidence), sampling_stride=stride,
                valid_image_regions=regions.tolist(), padding_excluded=True,
                reconstruction_to_world=alignment.matrix4().tolist(),
                pose_stage='post_ba' if ba_applied else 'vggt', dense_depth_refined=False,
                method='VGGT depth + predicted K + final camera extrinsics + calibration alignment',
                warning='Visualization estimate, not ground truth. BA refines poses/tracks, not dense depth.')


def preserve_offline_cloud(result, working_dir, destination):
    """Keep the cloud before the offline temporary directory is removed."""
    metadata = result.get('point_cloud')
    if not metadata: return
    source = Path(working_dir) / metadata['file']
    name = f'{Path(destination).stem}.point_cloud.{metadata["sha256"][:12]}.npz'
    target = Path(destination).with_name(name)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != metadata['sha256']:
            raise ValueError(f'Existing point-cloud file conflicts: {target}')
    else:
        shutil.copyfile(source, target)
    metadata['file'] = name


def load_cloud_buffer(calibration_path, metadata):
    """Validate a paired archive; return little-endian XYZ RGB confidence source float32."""
    base = Path(calibration_path).resolve().parent
    name = metadata.get('file', '')
    path = (base / name).resolve()
    if not name or Path(name).name != name or path.parent != base or path.suffix != '.npz':
        raise ValueError('Point cloud must be a sibling NPZ of the calibration JSON.')
    if metadata.get('coordinate_system') != WORLD_FRAME:
        raise ValueError('Unsupported point-cloud coordinate system.')
    if not path.is_file(): raise ValueError(f'Missing paired point cloud: {name}')
    if path.stat().st_size > 128_000_000: raise ValueError('Point-cloud archive exceeds size limit.')
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get('sha256'):
        raise ValueError('Point cloud does not match calibration metadata (SHA-256).')
    # Check decompressed sizes before numpy allocates any arrays.
    import zipfile
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 160_000_000:
            raise ValueError('Expanded point-cloud archive exceeds size limit.')
    with np.load(path, allow_pickle=False) as cloud:
        xyz, rgb, conf, source = (cloud[key] for key in ('xyz', 'rgb', 'confidence', 'source_index'))
        n = len(xyz)
        if not 0 < n <= 2_000_000 or n != metadata.get('point_count'):
            raise ValueError('Invalid point count.')
        if xyz.shape != (n,3) or rgb.shape != (n,3) or conf.shape != (n,) or source.shape != (n,):
            raise ValueError('Invalid point-cloud array shapes.')
        result = np.column_stack((xyz, rgb/255., conf, source)).astype('<f4')
        if (not np.isfinite(result).all() or np.any(np.abs(xyz) >= 1e6)
                or np.any(rgb < 0) or np.any(rgb > 255)
                or np.any(source < 0) or np.any(source >= len(metadata['sources']))
                or np.any(source != np.floor(source))):
            raise ValueError('Invalid point-cloud values.')
    return result.tobytes()
