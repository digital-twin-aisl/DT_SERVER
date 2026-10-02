# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from apps.calibration_worker.domain.manual import (
    rigid, move_camera, display_intrinsic, project, evaluate, refine, export_calibration,
)
from apps.calibration_worker.manual_editor import create_app, Workbench
from apps.calibration_worker.domain.lens_render import inverse_lens_map
from apps.calibration_worker.domain.point_cloud import export_depth_cloud
from apps.calibration_worker.domain.alignment import SimilarityTransform


def camera():
    return dict(camera_id='camera/1', camera_matrix=[[500.,0,320],[0,510.,240],[0,0,1]],
                distortion_coefficients=[-.1,.02,.001,-.002,0], source_image_size=[640,480],
                camera_to_world=np.eye(4).tolist(), world_to_camera=np.eye(4).tolist(), position_m=[0,0,0])


class ManualGeometryTests(unittest.TestCase):
    def test_inverse_lens_map_roundtrip_distorted_pixel_centres_and_overscan(self):
        k=np.array([[100.,0,80],[0,110.,60],[0,0,1]])
        d=np.array([-.12,.02,.001,-.002,0])
        meta,rays=inverse_lens_map(k,d,160,120)
        self.assertEqual(rays.shape,(120,160,2));self.assertEqual(rays.dtype,np.dtype('<f4'))
        self.assertEqual(meta['invalid_fraction'],0.)
        selected=rays[::7,::9].reshape(-1,2)
        restored,_=cv2.projectPoints(np.c_[selected,np.ones(len(selected))],np.zeros(3),np.zeros(3),k,d)
        yy,xx=np.mgrid[:120:7,:160:9]
        np.testing.assert_allclose(restored.reshape(-1,2),np.c_[xx.ravel()+.5,yy.ravel()+.5],atol=2e-5)
        low=np.array(meta['ray_bounds'][:2]);high=np.array(meta['ray_bounds'][2:])
        self.assertTrue((rays>low).all() and (rays<high).all())
        self.assertGreater(meta['source_size'][0],160)

    def test_zero_distortion_rays_preserve_focal_length_principal_point_and_hw(self):
        k=np.array([[81.,0,60],[0,93.,45],[0,0,1]])
        meta,rays=inverse_lens_map(k,np.zeros(5),140,100)
        self.assertEqual([meta['width'],meta['height']],[140,100])
        self.assertEqual(meta['camera_matrix'],k.tolist())
        np.testing.assert_allclose(rays[24,39],[(39.5-60)/81,(24.5-45)/93],atol=1e-7)

    def test_noninvertible_lens_regions_masked_instead_of_pinhole_fallback(self):
        k=np.array([[80.,0,80],[0,80.,60],[0,0,1]])
        meta,rays=inverse_lens_map(k,np.array([-.4,0,0,0,0]),160,120)
        self.assertGreater(meta['invalid_fraction'],.01)
        self.assertTrue(np.any(rays==1e9))
        self.assertLessEqual(meta['max_roundtrip_error_px'],.05)

    def test_rigid_validation(self):
        np.testing.assert_array_equal(rigid(np.eye(4)), np.eye(4))
        for m in [np.ones((4,4)), np.eye(3), np.full((4,4),np.nan), np.diag([-1,1,1,1])]:
            with self.assertRaises(ValueError): rigid(m)

    def test_world_translation_camera_local_rotation_and_inverse(self):
        base = move_camera(np.eye(4), [80,5,17], [0,0,90])
        actual = move_camera(base, [1,2,3], [90,0,0])
        np.testing.assert_allclose(actual[:3,3], [81,7,20])
        np.testing.assert_allclose(actual[:3,:3], base[:3,:3] @ move_camera(np.eye(4),[0,0,0],[90,0,0])[:3,:3])
        np.testing.assert_allclose(np.linalg.inv(actual) @ actual, np.eye(4), atol=1e-12)

    def test_projection_front_behind_and_empty(self):
        k=np.array(camera()['camera_matrix'])
        pixels,valid=project(np.eye(4),[[0,0,5],[1,1,5],[0,0,-5]],k,np.zeros(5))
        np.testing.assert_allclose(pixels[:2],[[320,240],[420,342]])
        self.assertEqual(valid.tolist(),[True,True,False])
        self.assertEqual(project(np.eye(4),[],k,np.zeros(5))[0].shape,(0,2))

    def test_rectification_and_rescaling_match_opencv(self):
        c=camera(); new,zero,k,d=display_intrinsic(c,(1280,960),(640,480),'rectified')
        np.testing.assert_allclose(k[:2],np.array(c['camera_matrix'])[:2]*2)
        np.testing.assert_array_equal(zero,np.zeros(5))
        points=np.array([[1,2,7.],[-1,-2,8.]])
        raw,_=project(np.eye(4),points,k,d)
        corrected=cv2.undistortPoints(raw.reshape(-1,1,2),k,d,P=new).reshape(-1,2)
        expected,_=project(np.eye(4),points,new,zero)
        np.testing.assert_allclose(corrected,expected,atol=.001)

    def test_pinhole_does_not_undistort_again(self):
        c=camera();c['undistorted_camera_matrix']=[[450,0,300],[0,450,220],[0,0,1]]
        new,d,_,_=display_intrinsic(c,(640,480),(640,480),'pinhole')
        self.assertEqual(new[0,0],450);self.assertEqual(np.count_nonzero(d),0)

    def test_measurement_rmse_holdout_and_behind_camera(self):
        k=np.array(camera()['camera_matrix'])
        pairs=[dict(world=[0,0,5],image=[323,244],holdout=False),dict(world=[0,0,5],image=[320,250],holdout=True),dict(world=[0,0,-5],image=[0,0])]
        result=evaluate(np.eye(4),pairs,k,np.zeros(5))
        self.assertEqual(result['metrics']['fit']['rmse_px'],5.)
        self.assertEqual(result['metrics']['holdout']['rmse_px'],10.)
        self.assertEqual(result['metrics']['all']['count'],2)
        self.assertIsNone(result['points'][2]['projected'])

    def test_refinement_recovers_pose_and_does_not_fit_holdout(self):
        rng=np.random.default_rng(42);world=rng.uniform([-2,-1,5],[2,1,10],size=(16,3))
        k=np.array(camera()['camera_matrix']);d=np.array(camera()['distortion_coefficients'])
        truth=move_camera(np.eye(4),[.13,-.09,.1],[1.2,-1.4,.8])
        observed,_=project(truth,world,k,d)
        pairs=[dict(world=w.tolist(),image=xy.tolist(),holdout=i>=12) for i,(w,xy) in enumerate(zip(world,observed))]
        result=refine(np.eye(4),pairs,k,d)
        self.assertLess(result['after']['metrics']['fit']['rmse_px'],1e-4)
        self.assertLess(result['after']['metrics']['holdout']['rmse_px'],1e-4)
        np.testing.assert_allclose(result['camera_to_world'],truth,atol=1e-5)
        changed=deepcopy(pairs)
        for p in changed[12:]:p['image'][0]+=100
        result2=refine(np.eye(4),changed,k,d)
        np.testing.assert_allclose(result2['camera_to_world'],result['camera_to_world'],atol=1e-10)
        self.assertGreater(result2['after']['metrics']['holdout']['rmse_px'],99)

    def test_refinement_rejects_insufficient_and_collinear(self):
        k=np.array(camera()['camera_matrix'])
        with self.assertRaises(ValueError):refine(np.eye(4),[],k,np.zeros(5))
        pairs=[dict(world=[i,0,10],image=[i,1]) for i in range(6)]
        with self.assertRaisesRegex(ValueError,'collinear'):refine(np.eye(4),pairs,k,np.zeros(5))

    def test_export_preserves_source_intrinsics_and_updates_all_extrinsics(self):
        document=dict(cameras=[camera()],coordinate_convention={'world':'USD metres'})
        before=deepcopy(document);pose=move_camera(np.eye(4),[1,2,3],[4,5,6])
        result=export_calibration(document,{'camera/1':pose},'source-hash')
        self.assertEqual(document,before)
        self.assertEqual(result['cameras'][0]['camera_matrix'],camera()['camera_matrix'])
        self.assertEqual(result['cameras'][0]['distortion_coefficients'],camera()['distortion_coefficients'])
        np.testing.assert_allclose(np.array(result['cameras'][0]['world_to_camera']) @ pose,np.eye(4),atol=1e-12)
        self.assertEqual(result['cameras'][0]['position_m'],[1,2,3])
        self.assertFalse(result['manual_correction']['automatically_deployed'])
        with self.assertRaises(ValueError):export_calibration(document,{'unknown':pose},'hash')


class ManualApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.calibration=self.root/'calibration.json'
        self.calibration.write_text(json.dumps(dict(cameras=[camera()],coordinate_convention={'world':'USD metres'})))
        self.original=self.calibration.read_bytes()
        self.meta=patch.object(Workbench,'video_info',return_value=dict(width=640,height=480,fps=30,frames=2,duration=1/30))
        self.meta.start()
        self.client=TestClient(create_app(self.calibration,self.root,self.root,(640,480)))
        self.client.__enter__()
        self.body=dict(camera_id='camera/1',camera_to_world=np.eye(4).tolist(),pairs=[])

    def tearDown(self):
        self.client.__exit__(None,None,None);self.meta.stop()
        self.assertEqual(self.calibration.read_bytes(),self.original)
        self.temp.cleanup()

    def test_config_accepts_string_query_and_returns_source_hash(self):
        r=self.client.get('/api/config?dataset=1');self.assertEqual(r.status_code,200)
        self.assertEqual(len(r.json()['source_sha256']),64)
        self.assertEqual(self.client.get('/api/config?dataset=4').status_code,422)

    def test_paired_cloud_endpoint_and_legacy_missing_cloud(self):
        self.assertEqual(self.client.get('/api/point-cloud').status_code,422)
        predictions=dict(depth=np.ones((1,2,2)),depth_conf=np.ones((1,2,2))*3,
            images=np.ones((1,2,2,3)),intrinsics=np.eye(3)[None],extrinsics=np.eye(4)[None,:3])
        alignment=SimilarityTransform(1.,np.eye(3),np.zeros(3),np.ones(3,dtype=bool),np.zeros(3))
        meta=export_depth_cloud(self.root/'cloud.npz',predictions,alignment,[dict(kind='cctv',camera_id='camera/1')])
        paired=self.root/'paired.json';paired.write_text(json.dumps(dict(cameras=[camera()],point_cloud=meta)))
        with TestClient(create_app(paired,self.root,self.root)) as client:
            config=client.get('/api/config').json()
            response=client.get('/api/point-cloud')
            self.assertEqual(response.status_code,200);self.assertEqual(len(response.content),4*8*4)
            self.assertEqual(response.headers['x-calibration-sha256'],config['source_sha256'])
            self.assertEqual(config['point_cloud']['sha256'],meta['sha256'])

    def test_preview_rejects_bad_camera_transform_points_mode(self):
        self.assertEqual(self.client.post('/api/project',json=self.body).status_code,200)
        for update in [dict(camera_id='../../etc/passwd'),dict(camera_to_world=np.zeros((4,4)).tolist()),dict(mode='other'),dict(pairs=[dict(world=[1,2],image=None)]),dict(pairs=[dict(world=[0,0,5],image=[9999,0])])]:
            self.assertEqual(self.client.post('/api/project',json=dict(self.body,**update)).status_code,422)

    def test_export_download_only_and_wrong_project_rejected(self):
        config=self.client.get('/api/config').json()
        data=dict(source_sha256=config['source_sha256'],edits={'camera/1':move_camera(np.eye(4),[1,0,0],[0,0,0]).tolist()})
        response=self.client.post('/api/export',json=data)
        self.assertEqual(response.status_code,200);self.assertIn('attachment',response.headers['content-disposition'])
        self.assertEqual(response.json()['cameras'][0]['position_m'],[1,0,0])
        data['source_sha256']='wrong';self.assertEqual(self.client.post('/api/export',json=data).status_code,422)

    def test_cross_origin_host_and_asset_access_restricted(self):
        self.assertEqual(self.client.get('/api/config',headers={'Origin':'https://untrusted.invalid'}).status_code,403)
        self.assertEqual(self.client.get('/api/config',headers={'Sec-Fetch-Site':'cross-site'}).status_code,403)
        self.assertEqual(self.client.get('/api/config',headers={'Host':'untrusted.invalid'}).status_code,400)
        self.assertEqual(self.client.get('/assets/calibration.json').status_code,404)

    def test_frame_range_validation(self):
        self.assertEqual(self.client.get('/api/frame?camera_id=camera/1&index=2').status_code,422)
        self.assertEqual(self.client.get('/api/frame?camera_id=camera/1&index=-1').status_code,422)

    def test_lens_endpoint_matches_projection_geometry_in_all_image_modes(self):
        for mode in ('raw','rectified','pinhole'):
            response=self.client.get('/api/lens-model',params={'camera_id':'camera/1','dataset':1,'mode':mode})
            self.assertEqual(response.status_code,200)
            meta=json.loads(response.headers['x-lens-model'])
            self.assertEqual(len(response.content),640*480*2*4)
            preview=self.client.post('/api/project',json=dict(self.body,mode=mode)).json()
            self.assertEqual(meta['camera_matrix'],preview['camera_matrix'])
            self.assertEqual(meta['camera_id'],'camera/1')
            if mode!='raw':self.assertTrue(all(x==0 for x in meta['distortion_coefficients']))


if __name__=='__main__':unittest.main()
