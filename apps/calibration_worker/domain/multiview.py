# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Multi-view camera projection and DLT triangulation in metres.

Cameras come from a calibration result document (OpenCV K/distortion and a
world-to-camera extrinsic). Triangulated points are estimates, not ground truth.
"""
import cv2
import numpy as np


def cameras_from_json(document):
    result={}
    for record in document['cameras']:
        key=str(int(record['camera_id'].split('/')[-1]))
        extrinsic=np.array(record['world_to_camera'],float)[:3]
        result[key]=dict(k=np.array(record['camera_matrix'],float),d=np.array(record['distortion_coefficients'],float),
            e=extrinsic,rv=cv2.Rodrigues(extrinsic[:,:3])[0],center=np.array(record['camera_to_world'])[:3,3])
    return result


def project(points,camera):
    points=np.asarray(points,float).reshape(-1,3)
    xy=cv2.projectPoints(points,camera['rv'],camera['e'][:,3],camera['k'],camera['d'])[0].reshape(-1,2)
    depth=(points@camera['e'][:,:3].T+camera['e'][:,3])[:,2]
    return xy,depth


def ray(xy,camera):
    xy=np.asarray(xy,float).reshape(1,1,2)
    value=cv2.undistortPointsIter(xy,camera['k'],camera['d'],None,None,
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-9)).reshape(2)
    restored=cv2.projectPoints(np.r_[value,1.][None],np.zeros(3),np.zeros(3),camera['k'],camera['d'])[0].reshape(2)
    if not np.isfinite(value).all() or np.linalg.norm(restored-xy.reshape(2))>.5:return None
    return value


def dlt(observations,cameras):
    rows=[]
    for cid,xy,confidence in observations:
        uv=ray(xy,cameras[cid])
        if uv is None:continue
        e=cameras[cid]['e'];w=np.sqrt(max(.01,float(confidence)))
        rows += [w*(uv[0]*e[2]-e[0]),w*(uv[1]*e[2]-e[1])]
    if len(rows)<4:return None
    _,_,vt=np.linalg.svd(rows,full_matrices=False);v=vt[-1]
    if abs(v[3])<1e-8:return None
    xyz=v[:3]/v[3]
    return xyz if np.isfinite(xyz).all() else None
