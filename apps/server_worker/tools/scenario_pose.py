# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Real PoseNet inference at CSV-specified roots using recorded video heatmaps.

No RootNet, Re-ID, synthetic gait, network services, or deployment changes.
CSV supplies XY/identity/time, terrain + configured clearance supplies Z.
"""
from bisect import bisect_right
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO/'packages/dt_common/src')]
import numpy as np
import torch
import yaml

from apps.server_worker.tools.scenario_skeleton import read_tracks, ground_height
from apps.server_worker.tools.pose_view_retry import RetryConfig, heatmap_peaks, retry_once, compact_report
from apps.server_worker.src.pose.core.config import config as pose_cfg, update_config
from apps.server_worker.src.pose.models.pose_regression_net import PoseRegressionNet
from apps.server_worker.src.utils.scene_recording import SceneOutputRecorder
from apps.server_worker.domain.ground_filter import add_ground_filter_arguments, make_ground_filter
from dt_common.calibration.identity import calibration_digest
from dt_common.contracts.scene import PersonEntity, PoseOutput, RootOutput, SceneOutput
from dt_common.inference_codec import decode_frame
from dt_common.spatial.workspace import _points_in_polygon
from apps.server_worker.tools.scene_geometry import load_scenes, sha256, write_json

DEFAULT_SOURCE = REPO/'apps/server_worker/data/recordings/runs/root_B_pose_B_v2_20260928_run2'
DEFAULT_CHECKPOINT = REPO/'apps/server_worker/models/POC_posenet.pth.tar'


def sample_roots(tracks, timestamp, ground, height):
    people = []
    for track in tracks:
        if not track.times[0] <= timestamp <= track.times[-1]: continue
        xy = track.at(timestamp)
        root = np.r_[xy, ground_height(ground, xy) + height]
        index = max(0, bisect_right(track.times, timestamp) - 1)
        people.append({'id': track.identity, 'root': root, 'flags': track.flags[index]})
    if len({p['id'] for p in people}) != len(people): raise ValueError('Overlapping CSV track segments')
    return people


def choose_edge(root, scenes, previous=None):
    """Choose one four-camera rig; prefer AOI, then visibility, then stable owner."""
    candidates = []
    for edge, scene in scenes.items():
        views = int(scene.visible(np.asarray(root)[None], scene.cameras)[0])
        inside = bool(_points_in_polygon(np.asarray(root)[None, :2], scene.workspace.footprint_xy_mm)[0])
        if views >= 2: candidates.append((int(inside), views, edge == previous, edge))
    if not candidates: return None, {'reason': 'fewer_than_two_visible_root_views'}
    inside, views, _, edge = max(candidates)
    return edge, {'visible_root_views': views, 'inside_edge_aoi': bool(inside)}


def anchor_pose(prediction, root):
    prediction = np.asarray(prediction, dtype=np.float64)
    root = np.asarray(root, dtype=np.float64)
    if prediction.shape != (15, 3) or root.shape != (3,) or not np.isfinite(prediction).all() or not np.isfinite(root).all():
        raise ValueError('Invalid pose or CSV root')
    shift = root - prediction[2]
    result = prediction + shift
    result[2] = root  # Exact externally supplied pelvis, not a new prediction.
    return result, shift


def packet_heatmaps(frame, scene, index, timestamp):
    # Deliberately do not read frame['roots']; externally supplied roots are authoritative.
    if frame['edge_id'] != scene.edge_id or frame['sequence'] != index:
        raise ValueError('Wrong cached edge/sequence')
    if frame['timestamp_kind'] != 'dataset_relative' or abs(frame['time'] - timestamp) > 1e-6:
        raise ValueError('Cached frame timestamp mismatch')
    if frame['camera_ids'] != [int(c['id']) for c in scene.cameras]: raise ValueError('Wrong camera order')
    if frame['calibration_digest'] != calibration_digest(scene.cameras): raise ValueError('Cached calibration differs')
    if frame['frame_ids'] != [index]*4 or any(abs(t-timestamp)>1e-6 for t in frame['frame_timestamps']):
        raise ValueError('Unsynchronized cached camera frames')
    heatmaps = frame['allheatmaps']
    expected = (1, 15, int(scene.heatmap_size[1]), int(scene.heatmap_size[0]))
    if len(heatmaps) != 4 or any(h.shape != expected or h.dtype != np.uint8 for h in heatmaps):
        raise ValueError('Expected four recorded uint8 15-joint heatmaps')
    return [torch.from_numpy(h.astype(np.float32) / 255.0) for h in heatmaps]


class PoseOnly:
    def __init__(self, checkpoint, cfg, scenes, device):
        source = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = {k.removeprefix('pose_net.'): v for k,v in source.items() if k.startswith('pose_net.')}
        if not state: raise ValueError('No PoseNet weights')
        self.models = {}; self.peaks = {}; self.device = device
        for edge, scene in scenes.items():
            transform = scene.transform * (np.asarray(cfg.NETWORK.IMAGE_SIZE) / scene.heatmap_size)[:, None]
            model = PoseRegressionNet(cfg, transform, scene.cameras).to(device).eval()
            model.load_state_dict(state, strict=True)
            def capture(module, inputs, output, edge=edge):
                self.peaks[edge] = output[0].amax(dim=(1,2,3,4)).detach().cpu().numpy()
            model.project_layer.register_forward_hook(capture)
            self.models[edge] = model

    def predict(self, edge, heatmaps, roots, excluded_view=None):
        centers = torch.as_tensor(np.asarray(roots), dtype=torch.float32, device=self.device)
        indices = torch.zeros(len(roots), dtype=torch.long, device=self.device)
        model = self.models[edge]
        cameras = model.project_layer.cams
        if len(heatmaps) != len(cameras): raise ValueError('Camera/heatmap count mismatch')
        if excluded_view is not None and not 0 <= excluded_view < len(cameras):
            raise ValueError('Invalid excluded view index')
        keep = [i for i in range(len(cameras)) if i != excluded_view]
        # This runner is synchronous. Restore even on exceptions so exclusion
        # cannot leak into subsequent people/frames or change camera ordering.
        try:
            model.project_layer.cams = [cameras[i] for i in keep]
            with torch.inference_mode():
                pred = model([heatmaps[i].to(self.device) for i in keep], centers, indices).cpu().numpy()
        finally:
            model.project_layer.cams = cameras
        if not np.isfinite(pred).all(): raise ValueError('Nonfinite PoseNet output')
        return pred, self.peaks[edge]


def prepare(args):
    plan = json.loads((args.source_run/'plan.json').read_text())
    completed = json.loads((args.source_run/'completed.json').read_text())
    deployment = args.source_run/'deployment.json'
    if sha256(deployment) != plan['deployment_sha256']: raise ValueError('Source deployment changed')
    manifest = json.loads(deployment.read_text())
    calibration = Path(manifest['calibration_result'])
    if sha256(calibration) != plan['calibration_sha256']: raise ValueError('Source calibration changed')
    runtime = Path(yaml.safe_load((args.source_run/'server.focus.yaml').read_text())['POSENET']['CONFIG'])
    scenes, context = load_scenes({'deployment': str(deployment), 'calibration': str(calibration), 'runtime_config': str(runtime)})
    datasets = {}
    for number in args.scenarios:
        key = f'0812_{number}'; csv = args.csv_dir/f'scenario_{number}.csv'
        tracks = read_tracks(csv, args.max_gap)
        frames = completed['results'][key]['frames']; fps = plan['datasets'][key]['fps']
        paths = {e: sorted((args.source_run/key/'replay'/e).glob('*.dtframe')) for e in scenes}
        if any(len(v) != frames or any(p.name != f'{i:09d}.dtframe' for i,p in enumerate(v)) for v in paths.values()):
            raise ValueError('Cached replay interval incomplete')
        if min(t.times[0] for t in tracks)+args.time_offset > (frames-1)/fps or max(t.times[-1] for t in tracks)+args.time_offset < 0:
            raise ValueError('CSV and video times do not overlap; supply explicit --time-offset')
        coverage = Counter()
        for track in tracks:
            for xy in track.xy:
                root = np.r_[xy, ground_height(context.ground_surface, xy)+args.root_height_mm]
                edge, _ = choose_edge(root, scenes)
                coverage[edge or 'no_visible_edge'] += 1
        datasets[number] = dict(key=key, csv=csv, tracks=tracks, frames=frames, fps=fps, paths=paths,
                                coverage=dict(coverage))
        print(json.dumps({'scenario': number, 'dataset': key, 'frames': frames, 'csv_ids': sorted({t.identity for t in tracks}),
                          'csv_start': min(t.times[0] for t in tracks), 'csv_end': max(t.times[-1] for t in tracks),
                          'csv_row_geometric_coverage': dict(coverage)}), flush=True)
    return plan, datasets, scenes, context, runtime


def run(args):
    torch.set_num_threads(4)
    plan, datasets, scenes, context, runtime = prepare(args)
    ground_filter = make_ground_filter(args, context.ground_surface)
    if args.preflight_only: return
    outputs = {n: REPO/'apps/server_worker/data/recordings'/f'{args.output_prefix}_0812_{n}.jsonl' for n in datasets}
    for path in outputs.values():
        if path.exists(): raise FileExistsError(path)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config(str(runtime))
    model = PoseOnly(args.checkpoint, pose_cfg, scenes, device)
    retry_config = RetryConfig(args.view_error_threshold, args.view_peak_confidence, args.view_min_joints) if args.view_retry else None
    provenance = {'mode': 'scenario_root_posenet', 'synthetic_pose': False,
        'root_source': 'CSV XY + terrain height + constant pelvis clearance', 'pose_source': 'real_video_heatmaps/PoseRegressionNet',
        'rootnet_used': False, 'reid_used': False, 'root_height_mm': args.root_height_mm,
        'time_offset_seconds': args.time_offset, 'time_mapping': 'video_time = csv_time + offset',
        'max_csv_gap_seconds': args.max_gap, 'pose_postprocess': 'translate all joints so pelvis equals external XYZ root',
        'checkpoint_sha256': sha256(args.checkpoint), 'calibration_sha256': plan['calibration_sha256'],
        'source_run': str(args.source_run), 'coordinate_system': 'USD world Z-up mm',
        'selection': 'all CSV people; zone/rank flags preserved but not applied',
        'edge_policy': 'at least 2 visible root views; prefer AOI then visible views then previous edge',
        'runtime_config_sha256': sha256(runtime), 'script_sha256': sha256(__file__),
        'view_retry': None if retry_config is None else retry_config.describe(),
        'view_retry_code_sha256': sha256(Path(__file__).with_name('pose_view_retry.py'))}
    write_json(args.run_dir/'plan.json', dict(provenance, checkpoint=str(args.checkpoint),
        scenarios={str(n): {'csv': str(d['csv']), 'csv_sha256': sha256(d['csv']), 'dataset': d['key']} for n,d in datasets.items()}))
    results = {}; started = time.monotonic()
    for number, data in datasets.items():
        folder = args.run_dir/f'scenario_{number}'; folder.mkdir()
        first = args.start_frame; end = min(data['frames'], first+args.max_frames if args.max_frames else data['frames'])
        if not 0 <= first < end: raise ValueError('Requested frame interval is empty')
        counts = Counter(); shifts = []; owners = {}; input_hash = hashlib.sha256(); comparisons = []
        info = dict(provenance, csv_sha256=sha256(data['csv']), csv=str(data['csv']), dataset=data['key'])
        with SceneOutputRecorder(outputs[number]) as recorder, (folder/'raw_predictions.jsonl').open('x') as diagnostics:
            for index in range(first, end):
                timestamp = index / data['fps']; csv_time = timestamp - args.time_offset
                people = sample_roots(data['tracks'], csv_time, context.ground_surface, args.root_height_mm)
                heatmaps = {}
                for edge in scenes:
                    payload = data['paths'][edge][index].read_bytes()
                    input_hash.update(edge.encode()); input_hash.update(index.to_bytes(8, 'big')); input_hash.update(payload)
                    heatmaps[edge] = packet_heatmaps(decode_frame(payload), scenes[edge], index, timestamp)
                assigned = {edge: [] for edge in scenes}
                for person in people:
                    edge, quality = choose_edge(person['root'], scenes, owners.get(person['id']))
                    person.update(edge=edge, quality=quality, pose=None, shift=None, cube_peak=None)
                    if edge is not None:
                        assigned[edge].append(person); owners[person['id']] = edge
                for edge, targets in assigned.items():
                    if not targets: continue
                    predictions, peaks = model.predict(edge, heatmaps[edge], [p['root'] for p in targets])
                    evidence = heatmap_peaks(heatmaps[edge], retry_config.peak_confidence) if retry_config else None
                    for person, raw, peak in zip(targets, predictions, peaks):
                        retry_report = None
                        if retry_config:
                            raw, peak, retry_report = retry_once(model.predict, anchor_pose, edge, heatmaps[edge],
                                scenes[edge], person['root'], raw, peak, evidence, retry_config)
                            person['quality']['view_retry'] = compact_report(retry_report)
                            counts['retry_attempts'] += retry_report['retry_count']
                            counts['retry_used'] += retry_report['retry_used']
                            counts['retry_reason_' + retry_report['reason']] += 1
                            if retry_report['retry_count']:
                                counts[f"excluded_camera_{retry_report['excluded_camera_id']}"] += 1
                            if 'retained_comparison' in retry_report:
                                comparison = retry_report['retained_comparison']
                                comparisons.append(comparison)
                                counts['retry_retained_error_improved'] += comparison['retry_mean_px'] < comparison['initial_mean_px']
                        person['cube_peak'] = float(peak)
                        if peak <= 0:
                            person['quality']['reason'] = 'empty_projected_heatmap_cube'
                        else:
                            person['pose'], person['shift'] = anchor_pose(raw, person['root'])
                            shifts.append(float(np.linalg.norm(person['shift'])))
                        diagnostics.write(json.dumps({'frame': index, 'timestamp': timestamp, 'global_id': person['id'],
                            'edge': edge, 'root_mm': person['root'].tolist(), 'raw_joints_mm': raw.tolist(),
                            'anchor_translation_mm': None if person['shift'] is None else person['shift'].tolist(),
                            'cube_peak': float(peak), 'view_retry': retry_report}, allow_nan=False)+'\n')
                entities = []; objects = []
                for candidate, person in enumerate(people):
                    has_pose = person['pose'] is not None
                    obj = PersonEntity(person['id'], 2 if has_pose else 1,
                        RootOutput(person['edge'] or 'scenario', candidate, person['root'].tolist(), 1.0, timestamp),
                        PoseOutput('voxelpose_15j_xyz', person['pose'].tolist()) if has_pose else None)
                    objects.append(obj); entity = obj.to_dict()
                    entity['scenario'] = dict(person['flags'], root_source='csv', pose_source='posenet' if has_pose else 'unavailable',
                        synthetic_pose=False, root_confidence_meaning='external constraint, not detector confidence',
                        cube_peak=person['cube_peak'], **person['quality'],
                        anchor_translation_mm=None if person['shift'] is None else person['shift'].tolist())
                    entities.append(entity)
                    counts['csv_person_observations'] += 1; counts['pose_observations'] += has_pose
                    counts['unavailable_pose_observations'] += not has_pose
                    if has_pose and not np.array_equal(person['pose'][2], person['root']): raise AssertionError('Pelvis not anchored')
                kept, ground_report = ground_filter.filter_people(objects)
                keep_ids = {p.global_id for p in kept}
                entities = [p for p in entities if p['global_id'] in keep_ids]
                counts['ground_rejected'] += ground_report['rejected_people']
                counts['output_pose_observations'] += sum(p['pose'] is not None for p in entities)
                scene = SceneOutput(timestamp, 0.0, [], runtime={'timestamp_kind': 'dataset_relative', 'scenario': info,
                    'ground_filter': ground_report}).to_dict()
                scene['people'] = entities
                recorder.write_payload(json.dumps(scene, allow_nan=False).encode())
                counts['frames'] += 1; counts['nonempty_frames'] += bool(entities)
                counts['frames_with_pose'] += any(p['pose'] is not None for p in entities)
                if index == first or (index+1)%200 == 0:
                    print(f"scenario_{number}: frame={index+1}/{end}, csv_people={len(people)}, pose_observations={counts['pose_observations']}", flush=True)
        summary = dict(counts, output=str(outputs[number]), output_sha256=sha256(outputs[number]),
            input_packets_sha256=input_hash.hexdigest(), first_frame=first, last_frame=end-1,
            fps=data['fps'], unique_csv_ids=sorted({t.identity for t in data['tracks']}),
            raw_pelvis_shift_mm=None if not shifts else {'median': float(np.median(shifts)), 'p95': float(np.quantile(shifts,.95)), 'max': max(shifts)},
            retained_reprojection_comparison=None if not comparisons else {
                'person_observations': len(comparisons),
                'initial_mean_px': float(np.mean([c['initial_mean_px'] for c in comparisons])),
                'retry_mean_px': float(np.mean([c['retry_mean_px'] for c in comparisons])),
                'note': 'Paired retained-view joints, equal person weighting; not ground-truth accuracy.'},
            note='Shift is the anchoring correction, not ground-truth pose error; XY/time/Z assumptions may be wrong.')
        write_json(folder/'result.json', summary); results[str(number)] = summary
        print('COMPLETED '+json.dumps(summary), flush=True)
    write_json(args.run_dir/'completed.json', {'results': results, 'elapsed_seconds': time.monotonic()-started})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_ground_filter_arguments(p)
    p.add_argument('--source-run', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--csv-dir', type=Path, default=REPO/'scenario (1)')
    p.add_argument('--scenarios', nargs='+', type=int, choices=[1,2,3], default=[1,2,3])
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-prefix', default='scenario_root_posenet_B_v2')
    p.add_argument('--root-height-mm', type=float, default=900.)
    p.add_argument('--time-offset', type=float, default=0.)
    p.add_argument('--max-gap', type=float, default=1.01)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--view-retry', action='store_true', help='Exclude the worst view and retry each person at most once')
    p.add_argument('--view-error-threshold', type=float, default=6., help='Median error threshold in heatmap pixels')
    p.add_argument('--view-peak-confidence', type=float, default=.1)
    p.add_argument('--view-min-joints', type=int, default=6)
    a = p.parse_args()
    try: RetryConfig(a.view_error_threshold, a.view_peak_confidence, a.view_min_joints)
    except ValueError as error: p.error(str(error))
    if not all(math.isfinite(v) for v in (a.root_height_mm,a.time_offset,a.max_gap)) or min(a.root_height_mm,a.max_gap)<=0:
        p.error('Invalid height, offset or gap')
    if a.start_frame<0 or (a.max_frames is not None and a.max_frames<1): p.error('Invalid frame interval')
    if len(set(a.scenarios))!=len(a.scenarios): p.error('Duplicate scenario')
    if not a.output_prefix or Path(a.output_prefix).name != a.output_prefix: p.error('Prefix must not be a path')
    for name in ('source_run','checkpoint','csv_dir','run_dir'): setattr(a,name,getattr(a,name).resolve())
    run(a)


if __name__ == '__main__': main()
