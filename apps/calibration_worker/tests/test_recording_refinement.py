# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
import unittest
import cv2
import numpy as np
from apps.calibration_worker.refine_recordings import evaluate,triangulate_track,undistort,no_median_regression


class RecordingRefinementTests(unittest.TestCase):
    def test_numerical_equality_is_not_a_regression(self):
        self.assertTrue(no_median_regression(12.445876948966735,12.445876948967973))
        self.assertTrue(no_median_regression(12.,11.))
        self.assertFalse(no_median_regression(12.,12.00001))

    def cameras(self):
        k=np.array([[1000.,0.,960.],[0.,1000.,540.],[0.,0.,1.]])
        return {str(i):dict(k=k,d=np.zeros(5),e=np.c_[np.eye(3),-np.array(center)],rv=np.zeros(3),center=np.array(center))
                for i,center in enumerate([[-1.,0.,0.],[1.,0.,0.],[0.,1.,0.]],1)}

    def test_exact_multiview_track_and_independent_reprojection(self):
        cams=self.cameras();point=np.array([.2,.3,5.])
        track=dict(observations={cid:cv2.projectPoints(point[None],c['rv'],c['e'][:,3],c['k'],c['d'])[0].reshape(2).tolist() for cid,c in cams.items()})
        np.testing.assert_allclose(triangulate_track(track,cams),point,atol=1e-8)
        self.assertLess(evaluate([track],cams)['overall']['median_px'],1e-8)
        cams['2']['e'][0,3]+=.2
        self.assertGreater(evaluate([track],cams)['overall']['median_px'],1.)

    def test_distortion_is_inverted_in_correct_pixel_coordinates(self):
        c=self.cameras()['1'];c['d']=np.array([-.2,.05,.001,-.002,0.])
        points=np.array([[.2,.3,5.],[1.,-1.,4.]])
        raw=cv2.projectPoints(points,c['rv'],c['e'][:,3],c['k'],c['d'])[0].reshape(-1,2)
        ideal=cv2.projectPoints(points,c['rv'],c['e'][:,3],c['k'],np.zeros(5))[0].reshape(-1,2)
        np.testing.assert_allclose(undistort(raw,c),ideal,atol=1e-6)

    def test_underconstrained_track_is_not_silently_accepted(self):
        cams=self.cameras();track=dict(observations={'1':[900.,500.]})
        self.assertIsNone(triangulate_track(track,cams))
        self.assertEqual(evaluate([track],cams)['invalid_tracks'],1)


if __name__=='__main__':unittest.main()
