# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
import json
from pathlib import Path
import tempfile
import unittest

import torch

from apps.deployments.rootnet_v2.record_datasets import audit_scene, combine_weights, aligned_packets


class RecordingBatchTests(unittest.TestCase):
    def test_decoded_common_interval_retains_original_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder); packets = p/'packets'
            for edge, count in {'edge_1': 3, 'edge_2': 4}.items():
                (packets/edge).mkdir(parents=True)
                for i in range(count): (packets/edge/f'{i:09d}.dtframe').write_bytes(b'packet')
            self.assertEqual(aligned_packets(packets, p/'replay', {'edge_1': 3, 'edge_2': 4}), 3)
            self.assertEqual(len(list((p/'replay/edge_2').glob('*.dtframe'))), 3)
            self.assertTrue((packets/'edge_2/000000003.dtframe').exists())
            self.assertEqual((packets/'edge_1/000000000.dtframe').stat().st_ino,
                             (p/'replay/edge_1/000000000.dtframe').stat().st_ino)

    def test_combination_uses_trained_root_and_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            root = {'backbone.x': torch.ones(1), 'root_net.x': torch.tensor([2.]), 'pose_net.x': torch.tensor([3.])}
            pose = {'backbone.x': torch.ones(1), 'root_net.x': torch.tensor([4.]), 'pose_net.x': torch.tensor([5.])}
            torch.save(root, p/'root.pt'); torch.save(pose, p/'pose.pt')
            combine_weights(p/'root.pt', p/'pose.pt', p/'combined.pt')
            result = torch.load(p/'combined.pt', weights_only=True)
            self.assertTrue(torch.equal(result['root_net.x'], root['root_net.x']))
            self.assertTrue(torch.equal(result['pose_net.x'], pose['pose_net.x']))
            with self.assertRaises(FileExistsError): combine_weights(p/'root.pt', p/'pose.pt', p/'combined.pt')

    def test_different_backbone_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            torch.save({'backbone.x': torch.ones(1)}, p/'root.pt')
            torch.save({'backbone.x': torch.zeros(1)}, p/'pose.pt')
            with self.assertRaisesRegex(ValueError, 'Backbones differ'):
                combine_weights(p/'root.pt', p/'pose.pt', p/'out.pt')
            self.assertFalse((p/'out.pt').exists())

    def test_empty_scenes_retained_and_frame_gaps_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'scenes.jsonl'
            rows = [{'schema_version': 1, 'coordinate_system': {'unit': 'millimetre'},
                     'timestamp': i/30, 'sync_spread_seconds': 0, 'people': []} for i in range(3)]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result = audit_scene(p, 3, 30)
            self.assertEqual(result['frames'], 3); self.assertEqual(result['nonempty_frames'], 0)
            with self.assertRaisesRegex(ValueError, 'Expected 4'): audit_scene(p, 4, 30)
            rows[1]['timestamp'] = 0.1
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'Missing/out-of-order'): audit_scene(p, 3, 30)

    def test_pose_counts_and_finite_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'scenes.jsonl'
            person = {'global_id': 1, 'root': {'position': [1., 2., 3.]}, 'pose': {'joints': [[1.,2.,3.]]*15}}
            scene = {'schema_version': 1, 'coordinate_system': {'unit': 'millimetre'},
                     'timestamp': 0, 'sync_spread_seconds': 0, 'people': [person]}
            p.write_text(json.dumps(scene)+'\n')
            result = audit_scene(p, 1, 30)
            self.assertEqual(result['pose_observations'], 1)
            person['pose']['joints'][0] = [float('nan'), 2., 3.]
            p.write_text(json.dumps(scene)+'\n')
            with self.assertRaises(ValueError): audit_scene(p, 1, 30)


if __name__ == '__main__': unittest.main()
