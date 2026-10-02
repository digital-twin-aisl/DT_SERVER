# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from apps.calibration_worker.domain.alignment import SimilarityTransform, transform_camera_extrinsic
from apps.calibration_worker.domain.point_cloud import export_depth_cloud, preserve_offline_cloud, load_cloud_buffer


def fixture():
    predictions = dict(depth=np.full((2,3,4),2.), depth_conf=np.full((2,3,4),3.),
        images=np.full((2,3,4,3),.5), intrinsics=np.tile(np.eye(3),(2,1,1)),
        extrinsics=np.tile(np.eye(4)[:3],(2,1,1)))
    predictions['extrinsics'][1,0,3] = -5.
    alignment=SimilarityTransform(2.,np.eye(3),np.array([10.,20.,30.]),np.ones(3,dtype=bool),np.zeros(3))
    sources=[dict(kind='cctv',camera_id='camera/1'),dict(kind='reference',frame_index=20)]
    return predictions,alignment,sources


class CloudTests(unittest.TestCase):
    def test_padding_is_excluded_without_removing_white_scene_pixels(self):
        predictions,alignment,sources=fixture()
        predictions['valid_regions']=np.array([[1,1,3,3],[0,0,4,3]])
        predictions['images'][:]=1.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cloud.npz'
            meta=export_depth_cloud(path,predictions,alignment,sources)
            self.assertEqual(meta['point_count'],16);self.assertTrue(meta['padding_excluded'])
            with np.load(path) as cloud:
                self.assertTrue((cloud['rgb']==255).all())
                self.assertEqual((cloud['source_index']==0).sum(),4)

    def test_preprocessing_regions_match_upstream_shapes(self):
        from apps.calibration_worker.inference import build_vggt_preprocessed_intrinsics
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i,(h,w) in enumerate([(1080,1920),(800,800)]):
                path=Path(tmp)/f'{i}.png';cv2.imwrite(str(path),np.zeros((h,w,3),np.uint8));paths.append(path)
            _,regions=build_vggt_preprocessed_intrinsics(paths,[np.eye(3)]*2,image_resolution=512,resize_mode='balanced',return_valid_regions=True)
            np.testing.assert_array_equal(regions,[[0,64,688,448],[88,0,600,512]])

    def test_rotated_metric_cloud_projects_with_final_camera(self):
        predictions,_,sources=fixture()
        rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        predictions['extrinsics'][0,:3,:3]=rotation
        predictions['extrinsics'][0,:3,3]=[2,3,4]
        alignment=SimilarityTransform(1.7,rotation,np.array([85.,5.,14.]),np.ones(3,dtype=bool),np.zeros(3))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cloud.npz'
            export_depth_cloud(path,predictions,alignment,sources)
            with np.load(path) as cloud:points=cloud['xyz'][:12]
            w2c,_=transform_camera_extrinsic(predictions['extrinsics'][0],alignment)
            camera_points=points @ w2c[:3,:3].T+w2c[:3,3]
            projected=camera_points[:,:2]/camera_points[:,2:3]
            yy,xx=np.mgrid[:3,:4]
            np.testing.assert_allclose(projected,np.c_[xx.ravel(),yy.ravel()],atol=2e-6)

    def test_offline_entrypoint_preserves_paired_artifact(self):
        from apps.calibration_worker import inference
        args=inference.build_argument_parser().parse_args([
            '--mode','offline','--images','photos','--reference-video','reference.mp4',
            '--checkpoint','weights.pt','--marker-tree','markers.usd',
            '--reference-sample-count','32','--max-images','40','--use_ba'])
        self.assertEqual(args.max_images,40)
        predictions,alignment,sources=fixture()
        def run(arguments):
            folder=Path(arguments.output_dir)
            meta=export_depth_cloud(folder/'point_cloud.npz',predictions,alignment,sources,ba_applied=arguments.use_ba)
            path=folder/'calibration_result.json'
            path.write_text(json.dumps(dict(input={},point_cloud=meta)))
            return path
        with tempfile.TemporaryDirectory() as tmp:
            args.json_output=str(Path(tmp)/'calibration.json')
            with patch.object(inference,'_interactive_marker_tree',return_value=Path('markers.usd')), \
                 patch.object(inference,'_resolve_checkpoint',return_value=Path('weights.pt')), \
                 patch.object(inference,'build_offline_manifest',return_value=(Path('manifest.json'),[])), \
                 patch.object(inference,'run_calibration',side_effect=run):
                path=inference.run_offline_calibration(args)
            self.assertFalse(Path(args.output_dir).exists())
            result=json.loads(path.read_text())
            self.assertEqual(result['input']['mode'],'offline_images')
            self.assertEqual(len(load_cloud_buffer(path,result['point_cloud'])),24*8*4)

    def test_final_pose_predicted_k_metric_alignment_and_color(self):
        predictions,alignment,sources=fixture()
        predictions['ba_intrinsics']=np.tile(np.diag([99.,99.,1.]),(2,1,1))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'point_cloud.npz'
            meta=export_depth_cloud(path,predictions,alignment,sources,ba_applied=True)
            with np.load(path) as cloud:
                np.testing.assert_allclose(cloud['xyz'][0],[10,20,34])
                np.testing.assert_allclose(cloud['xyz'][1],[14,20,34])
                np.testing.assert_allclose(cloud['xyz'][12],[20,20,34])
                np.testing.assert_array_equal(cloud['rgb'][0],[128]*3)
            data=np.frombuffer(load_cloud_buffer(Path(tmp)/'calibration.json',meta),dtype='<f4').reshape(-1,8)
            self.assertEqual(data.shape,(24,8));self.assertEqual(data[12,7],1)
            self.assertEqual(meta['pose_stage'],'post_ba');self.assertFalse(meta['dense_depth_refined'])

    def test_invalid_depth_confidence_and_point_budget(self):
        predictions,alignment,sources=fixture()
        predictions['depth'][0,0,0]=np.nan
        predictions['depth'][0,0,1]=-2
        predictions['depth_conf'][0,0,2]=.1
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cloud.npz'
            meta=export_depth_cloud(path,predictions,alignment,sources)
            self.assertEqual(meta['point_count'],21)
            meta=export_depth_cloud(path,predictions,alignment,sources,max_points=8)
            self.assertLessEqual(meta['point_count'],8)
            with np.load(path) as cloud:self.assertEqual(set(cloud['source_index']),{0,1})

    def test_offline_cloud_survives_temp_cleanup_and_hash_validation(self):
        predictions,alignment,sources=fixture()
        with tempfile.TemporaryDirectory() as final:
            destination=Path(final)/'calibration.json'
            with tempfile.TemporaryDirectory() as working:
                meta=export_depth_cloud(Path(working)/'point_cloud.npz',predictions,alignment,sources)
                result=dict(point_cloud=meta)
                preserve_offline_cloud(result,working,destination)
                destination.write_text(json.dumps(result))
            payload=load_cloud_buffer(destination,meta)
            self.assertEqual(len(payload),24*8*4)
            (destination.parent/meta['file']).write_bytes(b'wrong cloud')
            with self.assertRaisesRegex(ValueError,'SHA-256'):load_cloud_buffer(destination,meta)

    def test_reject_path_escape_and_wrong_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            for file in ['../cloud.npz','/tmp/cloud.npz','']:
                with self.assertRaises(ValueError):load_cloud_buffer(Path(tmp)/'calibration.json',dict(file=file))
            with self.assertRaises(ValueError):load_cloud_buffer(Path(tmp)/'calibration.json',dict(file='cloud.npz',coordinate_system='unknown'))


if __name__=='__main__':unittest.main()
