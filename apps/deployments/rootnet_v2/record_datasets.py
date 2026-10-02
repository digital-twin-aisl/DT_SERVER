# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Record actual edge RootNet + server PoseNet output, offline and frame-aligned.

Creates isolated profiles/checkpoint/logs; never changes existing launch profiles.
All output files are exclusive-create. Intermediate packets are retained for replay.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
PROFILE = Path(__file__).resolve().parent
RECORDINGS = REPO / 'apps/server_worker/data/recordings'
CAMERAS = {'edge_1': [2, 4, 6, 8], 'edge_2': [1, 3, 5, 7]}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def combine_weights(root_path, pose_path, output):
    root = torch.load(root_path, map_location='cpu', weights_only=True)
    pose = torch.load(pose_path, map_location='cpu', weights_only=True)
    if root.keys() != pose.keys(): raise ValueError('Full checkpoint keys differ')
    for key in root:
        if root[key].shape != pose[key].shape: raise ValueError(f'Shape mismatch: {key}')
        if not key.startswith(('backbone.', 'root_net.', 'pose_net.')):
            raise ValueError(f'Unexpected full-checkpoint key: {key}')
        if key.startswith('backbone.') and not torch.equal(root[key], pose[key]):
            raise ValueError('Backbones differ; refusing an unvalidated combination')
    merged = {k: pose[k] if k.startswith('pose_net.') else v for k, v in root.items()}
    if not all(torch.isfinite(v).all() for v in merged.values()):
        raise ValueError('Nonfinite model tensor')
    with output.open('xb') as stream: torch.save(merged, stream)
    return {'path': str(output), 'sha256': digest(output), 'tensors': len(merged),
            'root_source': str(root_path), 'root_source_sha256': digest(root_path),
            'pose_source': str(pose_path), 'pose_source_sha256': digest(pose_path),
            'backbone': 'bit-identical in both source exports',
            'root_tensors': sum(k.startswith('root_net.') for k in merged),
            'pose_tensors': sum(k.startswith('pose_net.') for k in merged)}


def probe(dataset):
    videos = []
    for edge, cameras in CAMERAS.items():
        for camera in cameras:
            path = REPO / f'apps/edge_client/data/data_{dataset}_{edge}/camera_{camera}.mkv'
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened(): raise ValueError(f'Cannot open {path}')
                item = {'path': str(path), 'edge': edge, 'camera': camera,
                        'frames': int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                        'fps': capture.get(cv2.CAP_PROP_FPS),
                        'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                if item['frames'] < 1 or item['fps'] <= 0 or not math.isfinite(item['fps']):
                    raise ValueError(f'Invalid video timing: {path}')
                videos.append(item)
            finally:
                capture.release()
    if max(v['fps'] for v in videos) - min(v['fps'] for v in videos) > 1e-6:
        raise ValueError('Video FPS differ; explicit resampling is required')
    if any((v['width'], v['height']) != (1920, 1080) for v in videos):
        raise ValueError('Video resolution differs from calibration')
    return {'videos': videos, 'fps': videos[0]['fps'],
            'common_frames': min(v['frames'] for v in videos)}


def run_process(command, log, env):
    print('RUN ' + ' '.join(map(str, command)), flush=True)
    with log.open('x') as stream:
        child = subprocess.Popen(list(map(str, command)), cwd=REPO, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    status = child.wait(timeout=20)
                    if status:
                        raise RuntimeError(f'Command failed ({status}); see {log}')
                    return
                except subprocess.TimeoutExpired:
                    with log.open('rb') as reader:
                        reader.seek(max(0, log.stat().st_size - 2048))
                        lines = reader.read().decode(errors='replace').splitlines()
                    print(f'{log.name}: {lines[-1] if lines else "starting"}', flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()


def audit_scene(path, expected_frames, fps):
    count = nonempty = pose_frames = people = poses = 0
    max_people = 0; ids = set(); first = last = None
    with path.open() as stream:
        for index, line in enumerate(stream):
            scene = json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
            if scene['schema_version'] != 1: raise ValueError('Wrong SceneOutput schema')
            if scene['coordinate_system']['unit'] != 'millimetre': raise ValueError('Wrong output units')
            stamp = scene['timestamp']
            if not math.isfinite(stamp) or abs(stamp - index / fps) > 1e-5:
                raise ValueError(f'Missing/out-of-order output frame {index}: {stamp}')
            if abs(scene['sync_spread_seconds']) > 1e-5: raise ValueError('Unaligned edge frames')
            if first is None: first = stamp
            last = stamp; count += 1
            entities = scene['people']; frame_poses = 0
            nonempty += bool(entities); people += len(entities); max_people = max(max_people, len(entities))
            for entity in entities:
                ids.add(entity['global_id'])
                if not all(math.isfinite(x) for x in entity['root']['position']): raise ValueError('Invalid root')
                if entity['pose'] is not None:
                    joints = entity['pose']['joints']
                    if len(joints) != 15 or any(len(j) != 3 or not all(math.isfinite(x) for x in j) for j in joints):
                        raise ValueError('Invalid 15-joint pose')
                    poses += 1; frame_poses += 1
            pose_frames += frame_poses > 0
    if count != expected_frames: raise ValueError(f'Expected {expected_frames} scenes, got {count}')
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size,
            'frames': count, 'nonempty_frames': nonempty, 'frames_with_pose': pose_frames,
            'person_observations': people, 'pose_observations': poses, 'max_people': max_people,
            'distinct_track_ids': len(ids), 'first_timestamp': first, 'last_timestamp': last,
            'clip_duration_seconds': count / fps, 'all_frames_aligned': True}


def aligned_packets(packet_root, replay_root, counts):
    """Keep originals; replay only the decoded interval available from both edges."""
    if not counts or any(n < 1 for n in counts.values()): raise ValueError('Empty edge recording')
    common = min(counts.values())
    replay_root.mkdir()
    for edge, count in counts.items():
        source = packet_root/edge
        packets = sorted(source.glob('*.dtframe'))
        if len(packets) != count or any(p.name != f'{i:09d}.dtframe' for i,p in enumerate(packets)):
            raise ValueError(f'Non-contiguous packet sequence: {edge}')
        target = replay_root/edge; target.mkdir()
        for packet in packets[:common]: os.link(packet, target/packet.name)
    return common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--datasets', nargs='+', choices=['0812_1', '0812_2', '0812_3'], default=['0812_1', '0812_2', '0812_3'])
    parser.add_argument('--output-prefix', default='rootnet_B_posenet_B_v2')
    parser.add_argument('--max-frames', type=int, help='Optional bounded smoke test; default entire common video interval')
    parser.add_argument('--calibration', type=Path, default=PROFILE/'calibration.from_cameras_v2.json')
    parser.add_argument('--root-checkpoint', type=Path, default=REPO/'SelfPose3d/output_root_robustness/pilot_v2/B_export.pth.tar')
    parser.add_argument('--pose-checkpoint', type=Path, default=REPO/'SelfPose3d/output_pose_synthetic/full_20260928/B_export.pth.tar')
    args = parser.parse_args()
    if Path(args.output_prefix).name != args.output_prefix or not args.output_prefix:
        parser.error('output-prefix must be a filename prefix, not a path')
    if len(set(args.datasets)) != len(args.datasets): parser.error('Duplicate dataset')
    if args.max_frames is not None and args.max_frames < 1: parser.error('max-frames must be positive')
    run = Path(args.run_dir).resolve()
    outputs = {d: RECORDINGS / f'{args.output_prefix}_{d}.jsonl' for d in args.datasets}
    for path in outputs.values():
        if path.exists(): raise FileExistsError(path)
    run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); cv2.setNumThreads(2)
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENCV_FOR_THREADS_NUM='2')
    env['PYTHONPATH'] = os.pathsep.join([str(REPO/'packages/dt_common/src'), str(REPO), env.get('PYTHONPATH', '')])
    deployment = json.loads((PROFILE/'deployment.json').read_text())
    for key in ('ground_usd', 'ground_cache'):
        if deployment['scene'].get(key): deployment['scene'][key] = str((PROFILE/deployment['scene'][key]).resolve())
    # Snapshot the explicit calibration so training/replay can share one immutable input.
    write_json(run/'calibration.json', json.loads(args.calibration.read_text()))
    deployment['calibration_result'] = str(run/'calibration.json')
    write_json(run/'deployment.json', deployment)
    weights = combine_weights(args.root_checkpoint.resolve(), args.pose_checkpoint.resolve(), run/'root_B_pose_B.pth.tar')
    for role in ('edge', 'server'):
        config = REPO / f'apps/{"edge_client" if role == "edge" else "server_worker"}/config/cam4_posenet.yaml'
        with (run/f'{role}.focus.yaml').open('x') as stream:
            yaml.safe_dump({'POSENET': {'CKPT': weights['path'], 'CONFIG': str(config), 'TENSORRT': False}}, stream)
    datasets = {d: probe(d) for d in args.datasets}
    plan = {'datasets': datasets, 'weights': weights, 'deployment_sha256': digest(run/'deployment.json'),
            'calibration_sha256': digest(deployment['calibration_result']),
            'input_mode': 'offline strict frame-aligned', 'reid': False, 'lod_policy': 'all',
            'max_frames_override': args.max_frames, 'existing_profiles_modified': False,
            'script_sha256': digest(__file__), 'python': sys.executable,
            'end_policy': 'Decode to EOF; use common actual packet count, not approximate MKV metadata; retain unmatched tails',
            'calibration_source': str(args.calibration.resolve()),
            'note': 'Explicit calibration snapshot; not ground-truth accuracy validation; IDs use root tracking fallback'}
    write_json(run/'plan.json', plan)
    results = {}; started = time.monotonic()
    for dataset, info in datasets.items():
        folder = run/dataset; folder.mkdir()
        counts = {}
        for edge in CAMERAS:
            command = [sys.executable, REPO/'apps/edge_client/inference.py',
                '--runtime-config', PROFILE/f'{edge}.json', '--deployment', run/'deployment.json',
                '--cfg-focus', run/'edge.focus.yaml', '--example-folder', REPO/f'apps/edge_client/data/data_{dataset}_{edge}',
                '--no-require-identity', '--no-zenoh', '--no-reid', '--no-tensorrt',
                '--record-output', folder/'packets']
            if args.max_frames is not None: command += ['--max-frames', args.max_frames]
            run_process(command, folder/f'{edge}.log', env)
            packets = sorted((folder/'packets'/edge).glob('*.dtframe'))
            if not packets or any(p.name != f'{i:09d}.dtframe' for i, p in enumerate(packets)):
                raise ValueError(f'{dataset}/{edge}: missing or non-contiguous packets')
            expected = min(v['frames'] for v in info['videos'] if v['edge'] == edge)
            if args.max_frames is not None: expected = min(expected, args.max_frames)
            # MKV frame count is commonly inferred from duration and is off by 1.
            # Large discrepancies may mean a corrupt/unreadable video: fail visibly.
            if abs(len(packets) - expected) > 5:
                raise ValueError(f'{dataset}/{edge}: metadata says {expected}, decoded {len(packets)}; inspect source before accepting')
            counts[edge] = len(packets)
        frames = aligned_packets(folder/'packets', folder/'replay', counts)
        alignment = {'decoded_edge_frames': counts, 'common_frames': frames,
                     'unpaired_tail_frames': {e: n-frames for e,n in counts.items()},
                     'metadata_common_frames': info['common_frames']}
        write_json(folder/'alignment.json', alignment)
        print('ALIGNMENT ' + json.dumps(alignment), flush=True)
        command = [sys.executable, REPO/'apps/server_worker/inference.py',
            '--runtime-config', PROFILE/'server.json', '--deployment', run/'deployment.json',
            '--cfg-focus', run/'server.focus.yaml', '--replay-inputs', folder/'replay',
            '--input-mode', 'strict', '--input-clock', 'dataset', '--no-tensorrt',
            '--no-scene-zenoh', '--no-zmq', '--scene-recording', outputs[dataset],
            '--metrics-dir', folder/'metrics', '--metrics-run-label', f'{args.output_prefix}_{dataset}']
        run_process(command, folder/'server.log', env)
        result = audit_scene(outputs[dataset], frames, info['fps'])
        write_json(folder/'result.json', result); results[dataset] = result
        print('COMPLETED ' + json.dumps(result), flush=True)
    write_json(run/'completed.json', {'results': results, 'elapsed_seconds': time.monotonic()-started})


if __name__ == '__main__': main()
