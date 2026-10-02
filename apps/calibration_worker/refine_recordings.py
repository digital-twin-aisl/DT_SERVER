# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Recording correspondences and bounded fixed-intrinsic camera refinement.

No network/model learning, no ground/pose outputs as calibration truth. Units m.
Original world gauge/metric baseline retained; this is relative refinement.
"""
import argparse
from collections import Counter
from copy import deepcopy
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'packages/dt_common/src')]
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from apps.server_worker.tools.scene_geometry import sha256,write_json
from apps.calibration_worker.domain.multiview import cameras_from_json,project,dlt

DEFAULT_CALIBRATION=REPO/'apps/calibration_worker/data/vggt_cloud_90.NWJ28r/calibration_result.json'
DEFAULT_EVIDENCE=REPO/'SelfPose3d/output_real_adaptation.woP2Wy/evidence_v2'
DEFAULT_LABELS=REPO/'SelfPose3d/output_real_adaptation.woP2Wy/labels_full90'


def no_median_regression(before,after):
    # Triangulation can differ at ~1e-12 px even for unchanged cameras.
    return after <= before + 1e-8


def mutual_matches(a,b,ratio=.72):
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=100))
    def selected(x,y):
        return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(x,y,k=2) if len(pair)==2
                for m,n in [pair] if m.distance<ratio*n.distance}
    ab,ba=selected(a,b),selected(b,a)
    return [(i,j) for i,j in ab.items() if ba.get(j)==i]


def undistort(points,camera):
    return cv2.undistortPointsIter(np.asarray(points,float)[:,None],camera['k'],camera['d'],None,camera['k'],
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-9)).reshape(-1,2)


def extract(args):
    cv2.setNumThreads(4);cv2.setRNGSeed(20260929)
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.evidence/'manifest.json').read_text())
    document=json.loads(args.calibration.read_text()); cameras=cameras_from_json(document)
    sift=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.02)
    features={}; drift={}; started=time.monotonic()
    for cid in sorted(cameras,key=int):
        captures=[]; medians=[]
        for dataset,info in sorted(manifest['datasets'].items()):
            video=next(v for v in info['videos'] if str(v['camera'])==cid)
            path=Path(video['path']); stat=path.stat()
            if stat.st_size!=video['bytes'] or stat.st_mtime_ns!=video['mtime_ns']:raise ValueError('Source video changed')
            cap=cv2.VideoCapture(str(path)); frames=[]
            try:
                for index in np.linspace(0,info['frames']-1,13).astype(int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,im=cap.read()
                    if not ok:raise ValueError(f'Cannot decode {path}/{index}')
                    frames.append(im)
            finally:cap.release()
            median=np.median(np.stack(frames),axis=0).astype(np.uint8)
            medians.append(median)
            gray=cv2.cvtColor(median,cv2.COLOR_BGR2GRAY)
            keypoints,desc=sift.detectAndCompute(gray,None)
            captures.append((np.float32([p.pt for p in keypoints]),desc))
        pts,desc=captures[0]; measures=[]
        for other_pts,other_desc in captures[1:]:
            matches=mutual_matches(desc,other_desc)
            xy1=np.asarray([pts[i] for i,j in matches]);xy2=np.asarray([other_pts[j] for i,j in matches])
            _,mask=cv2.findHomography(xy1,xy2,cv2.USAC_MAGSAC,2.,maxIters=10000,confidence=.999)
            if mask is None or mask.sum()<50:raise ValueError(f'Cannot verify static camera across recordings: {cid}')
            distances=np.linalg.norm(xy1-xy2,axis=1)[mask.ravel().astype(bool)]
            measures.append(dict(matches=len(distances),median_px=float(np.median(distances)),p95_px=float(np.quantile(distances,.95))))
            if np.median(distances)>3.:raise ValueError(f'Camera {cid} moved across recordings; separate calibrations required')
        drift[cid]=measures
        cv2.imwrite(str(args.output/f'background_{cid}.jpg'),medians[0])
        features[cid]=(pts,desc,undistort(pts,cameras[cid]))
        print('FEATURES',cid,len(pts),'same-camera drift',measures,flush=True)
    # Merge pair correspondences, disallow two different features of one camera.
    parent={}; members={}
    def find(x):
        if x not in parent:parent[x]=x;members[x]={x[0]:x[1]}
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    edges=[]; pair_reports={}
    for a,b in combinations(features,2):
        pa,da,ua=features[a];pb,db,ub=features[b]; matches=mutual_matches(da,db)
        if len(matches)<12:pair_reports[f'{a}-{b}']=dict(matches=len(matches),inliers=0);continue
        xa=np.array([ua[i] for i,j in matches]);xb=np.array([ub[j] for i,j in matches])
        _,mask=cv2.findFundamentalMat(xa,xb,cv2.USAC_MAGSAC,1.5,.9999,20000)
        good=[] if mask is None else [pair for pair,ok in zip(matches,mask.ravel()) if ok]
        pair_reports[f'{a}-{b}']=dict(matches=len(matches),inliers=len(good))
        if len(good)<12:continue
        for i,j in good:edges.append((a,i,b,j))
        print('PAIR',a,b,'mutual',len(matches),'inliers',len(good),flush=True)
    for a,i,b,j in edges:
        x,y=find((a,i)),find((b,j))
        if x==y:continue
        if any(c in members[y] and members[y][c]!=idx for c,idx in members[x].items()):continue
        parent[y]=x;members[x].update(members.pop(y))
    tracks=[]; seen=[]
    for group in members.values():
        if len(group)<2:continue
        obs=[(cid,features[cid][0][index],1.) for cid,index in sorted(group.items())]
        xyz=dlt(obs,cameras)
        if xyz is None:continue
        errors=[];valid=True
        for cid,xy,_ in obs:
            uv,depth=project(xyz,cameras[cid]);errors.append(float(np.linalg.norm(uv[0]-xy)))
            valid &= .3<float(depth[0])<100 and errors[-1]<100
        if not valid:continue
        rays=np.array([(xyz-cameras[cid]['center'])/np.linalg.norm(xyz-cameras[cid]['center']) for cid,_,_ in obs])
        if np.min(rays@rays.T)>.9994:continue
        # Deduplicate SIFT scale/orientation duplicates; split by physical feature.
        duplicate=False
        for previous in seen:
            common=set(group)&set(previous)
            if len(common)>=2 and all(np.linalg.norm(features[c][0][group[c]]-features[c][0][previous[c]])<3 for c in common):
                duplicate=True;break
        if duplicate:continue
        seen.append(group)
        tracks.append(dict(observations={cid:xy.tolist() for cid,xy,_ in obs},initial_xyz_m=xyz.tolist(),initial_error_px=errors))
    rng=np.random.default_rng(20260929);order=rng.permutation(len(tracks))
    for k,i in enumerate(order):tracks[i]['split']='holdout' if k%5==0 else 'train'
    counts={s:dict(Counter(cid for t in tracks if t['split']==s for cid in t['observations'])) for s in ['train','holdout']}
    report=dict(calibration=str(args.calibration.resolve()),calibration_sha256=sha256(args.calibration),
        evidence=str(args.evidence.resolve()),evidence_manifest_sha256=sha256(args.evidence/'manifest.json'),
        tracks=tracks,pairs=pair_reports,counts=counts,same_camera_cross_recording_drift=drift,
        method='13-frame median backgrounds; mutual SIFT ratio .72; undistorted fundamental RANSAC 1.5px; deduplicated 80/20 track split',
        coordinate_system='USD world Z-up metres',elapsed_seconds=time.monotonic()-started)
    write_json(args.output/'tracks.json',report)
    print('TRACKS',len(tracks),json.dumps(counts),flush=True)


def triangulate_track(track,cameras):
    observations=[(cid,np.asarray(xy),1.) for cid,xy in track['observations'].items()]
    return dlt(observations,cameras)


def augment(args):
    """Reuse only cross-view associations and raw independent YOLO 2D pixels."""
    data=json.loads(args.tracks.read_text());cameras=cameras_from_json(json.loads(Path(data['calibration']).read_text()))
    labels=args.labels
    old=json.loads((labels/'manifest.json').read_text())
    if sha256(labels/'samples.jsonl')!=old['samples_sha256']:raise ValueError('Association evidence changed')
    evidence=Path(data['evidence']);manifest=json.loads((evidence/'manifest.json').read_text())
    if sha256(evidence/'detections.jsonl')!=manifest['detections_sha256']:raise ValueError('2D evidence changed')
    rows={(r['dataset'],r['frame']):r for line in (evidence/'detections.jsonl').open() for r in [json.loads(line)]}
    candidates=[];seen=set()
    for line in (labels/'samples.jsonl').open():
        row=json.loads(line);key=(row['dataset'],row['frame']);raw=rows[key]
        if row['frame']%18:continue
        for person in row['people']:
            identity=(*key,tuple(sorted(person['members'].items())))
            if identity in seen:continue
            seen.add(identity)
            for joint in range(15):
                if joint in (0,2):continue  # Avoid virtual midpoint joints.
                obs={cid:raw['cameras'][cid][person['members'][cid]]['keypoints'][joint][:2]
                     for cid in person['joint_views'][joint]
                     if raw['cameras'][cid][person['members'][cid]]['keypoints'][joint][2]>=.55}
                if len(obs)<2:continue
                t=dict(observations=obs,kind='independent_2d_human',dataset=row['dataset'],frame=row['frame'],joint=joint,
                       split='holdout' if row['dataset']=='0812_3' else 'train')
                xyz=triangulate_track(t,cameras)
                if xyz is None:continue
                t['initial_xyz_m']=xyz.tolist();candidates.append(t)
    rng=np.random.default_rng(20260929)
    # Limit CPU solve size, balanced by recording; reserve ALL third-clip humans.
    selected=[]
    for dataset in ['0812_1','0812_2','0812_3']:
        group=[t for t in candidates if t['dataset']==dataset]
        order=list(rng.permutation(len(group)));counts=Counter();chosen=[]
        # Cover the weak even cameras instead of letting strong odd cameras dominate.
        while order and len(chosen)<800:
            cid=min(cameras,key=lambda c:counts[c])
            eligible=next((k for k,i in enumerate(order) if cid in group[i]['observations']),0)
            i=order.pop(eligible);chosen.append(group[i]);counts.update(group[i]['observations'].keys())
        selected.extend(chosen)
    for track in data['tracks']:track['kind']='static_background'
    data['tracks'].extend(selected)
    data['counts']={s:dict(Counter(cid for t in data['tracks'] if t['split']==s for cid in t['observations'])) for s in ['train','holdout']}
    data['human_evidence']=dict(labels_manifest_sha256=sha256(labels/'manifest.json'),detections_sha256=manifest['detections_sha256'],
        source='Original-calibration association/inlier seeds only; triangulated XYZ is initialization, never a fixed target',
        split='Humans: clips 0812_1+0812_2 fit, entire 0812_3 holdout; static tracks retain 80/20 split',
        candidate_tracks=len(candidates),selected_tracks=len(selected))
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'tracks.json',data)
    print('AUGMENTED',json.dumps(dict(counts=data['counts'],humans=data['human_evidence'])),flush=True)


def evaluate(tracks,cameras):
    errors=[]; per_camera={c:[] for c in cameras}; invalid=0
    for track in tracks:
        xyz=triangulate_track(track,cameras)
        if xyz is None:invalid+=1;continue
        for cid,xy in track['observations'].items():
            uv,depth=project(xyz,cameras[cid]);error=float(np.linalg.norm(uv[0]-xy)) if depth[0]>.1 else 10000.
            if not np.isfinite(error):error=10000.
            errors.append(error);per_camera[cid].append(error)
    def stats(values):
        return dict(observations=len(values),median_px=float(np.median(values)),p95_px=float(np.quantile(values,.95)),mean_px=float(np.mean(values))) if values else None
    return dict(overall=stats(errors),per_camera={c:stats(v) for c,v in per_camera.items()},invalid_tracks=invalid)


def refine(args):
    cv2.setNumThreads(2)
    data=json.loads(args.tracks.read_text());document=json.loads(Path(data['calibration']).read_text())
    if sha256(data['calibration'])!=data['calibration_sha256']:raise ValueError('Initial calibration changed')
    cameras=cameras_from_json(document);order=sorted(cameras,key=int);anchor=order[0];variable=args.refine_cameras
    if not variable or len(set(variable))!=len(variable) or any(c not in cameras or c==anchor for c in variable):
        raise ValueError('Invalid refinable camera list')
    training=[t for t in data['tracks'] if t['split']=='train'];holdout=[t for t in data['tracks'] if t['split']=='holdout']
    if args.fit_kind=='human':training=[t for t in training if t.get('kind')=='independent_2d_human']
    for cid in order:
        if data['counts']['train'].get(cid,0)<20 or data['counts']['holdout'].get(cid,0)<5:
            raise ValueError(f'Insufficient independent evidence for camera {cid}')
    args.output.mkdir(parents=True,exist_ok=False)
    origin=np.mean([c['center'] for c in cameras.values()],axis=0)
    points=np.array([t['initial_xyz_m'] for t in training])-origin
    ncam=len(variable);offset=6*ncam
    x0=np.r_[np.zeros(offset),points.ravel()]
    observations={cid:[] for cid in order}
    for i,t in enumerate(training):
        for cid,xy in t['observations'].items():observations[cid].append((i,xy))
    obs_count=sum(map(len,observations.values()))
    bounds_lo=np.r_[np.tile(np.r_[np.full(3,-np.deg2rad(8)),np.full(3,-.75)],ncam),np.full(points.size,-np.inf)]
    bounds_hi=-bounds_lo
    def unpack(x):
        result={}
        for cid,cam in cameras.items():
            delta=np.zeros(6) if cid not in variable else x[6*variable.index(cid):6*(variable.index(cid)+1)]
            rotation=cv2.Rodrigues(delta[:3])[0]@cam['e'][:,:3]
            center=cam['center']+(np.zeros(3) if args.rotation_only else delta[3:])
            result[cid]=dict(cam,center=center,e=np.c_[rotation,-rotation@center],rv=cv2.Rodrigues(rotation)[0])
        return result,x[offset:].reshape(-1,3)+origin
    baseline=np.linalg.norm(cameras['5']['center']-cameras[anchor]['center'])
    def residual(x):
        current,xyz=unpack(x); values=[]
        for cid,obs in observations.items():
            indices=np.array([o[0] for o in obs]);xy=np.array([o[1] for o in obs])
            uv,depth=project(xyz[indices],current[cid]);diff=uv-xy
            diff[depth<.1]=1000.;values.extend(diff.ravel())
        # Small zero-mean extrinsic priors, preserve world scale using a baseline.
        for i in range(ncam):
            d=x[i*6:(i+1)*6];values.extend(d[:3]/np.deg2rad(2));values.extend(d[3:]/.20)
        values.append((np.linalg.norm(current['5']['center']-current[anchor]['center'])-baseline)/.005)
        return np.asarray(values)
    sparsity=lil_matrix((2*obs_count+offset+1,len(x0)),dtype=int);row=0
    for cid,obs in observations.items():
        for i,_ in obs:
            sparsity[row:row+2,offset+3*i:offset+3*(i+1)]=1
            if cid in variable:
                k=variable.index(cid);sparsity[row:row+2,6*k:6*(k+1)]=1
            row+=2
    sparsity[row:row+offset,:offset]=np.eye(offset,dtype=int);row+=offset
    if '5' in variable:
        k=variable.index('5');sparsity[row,6*k+3:6*k+6]=1
    started=time.monotonic()
    before=evaluate(holdout,cameras);print('HOLDOUT BEFORE',json.dumps(before),flush=True)
    if args.solver=='epipolar':
        pairs={}
        for track in training:
            for a,b in combinations(sorted(track['observations']),2):
                if a not in variable and b not in variable:continue
                pairs.setdefault((a,b),[]).append((track['observations'][a],track['observations'][b]))
        prepared={}
        for (a,b),obs in pairs.items():
            uv=np.array(obs);pa=undistort(uv[:,0],cameras[a]);pb=undistort(uv[:,1],cameras[b])
            prepared[(a,b)]=(np.c_[pa,np.ones(len(pa))],np.c_[pb,np.ones(len(pb))])
        def epipolar(x):
            current,_=unpack(np.r_[x,points.ravel()]);values=[]
            for (a,b),(pa,pb) in prepared.items():
                ca,cb=current[a],current[b];relative=cb['e'][:,:3]@ca['e'][:,:3].T
                t=cb['e'][:,:3]@(ca['center']-cb['center'])
                cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0.]])
                fundamental=np.linalg.inv(cb['k']).T@cross@relative@np.linalg.inv(ca['k'])
                line_b=pa@fundamental.T;line_a=pb@fundamental
                numerator=np.sum(pb*line_b,axis=1)
                denominator=np.sqrt(np.sum(line_b[:,:2]**2+line_a[:,:2]**2,axis=1)+1e-20)
                values.extend(numerator/denominator)
            for i in range(ncam):
                d=x[i*6:(i+1)*6];values.extend(d[:3]/np.deg2rad(2));values.extend(d[3:]/.20)
            values.append((np.linalg.norm(current['5']['center']-current[anchor]['center'])-baseline)/.005)
            return np.asarray(values)
        solution=least_squares(epipolar,np.zeros(offset),jac='3-point',bounds=(bounds_lo[:offset],bounds_hi[:offset]),
            loss='huber',f_scale=2.,x_scale=np.tile([.02,.02,.02,.2,.2,.2],ncam),max_nfev=args.iterations,
            ftol=1e-7,xtol=1e-8,gtol=1e-5,verbose=1)
        solution.x=np.r_[solution.x,points.ravel()]
    else:
        solution=least_squares(residual,x0,jac='3-point',jac_sparsity=sparsity.tocsr(),bounds=(bounds_lo,bounds_hi),
            loss='huber',f_scale=2.,x_scale=np.r_[np.tile([.02,.02,.02,.2,.2,.2],ncam),np.ones(points.size)],
            max_nfev=args.iterations,ftol=1e-6,xtol=1e-7,gtol=1e-5,verbose=1)
    refined,_=unpack(solution.x);after=evaluate(holdout,refined)
    by_kind={kind:dict(before=evaluate([t for t in holdout if t.get('kind','static_background')==kind],cameras),
                      after=evaluate([t for t in holdout if t.get('kind','static_background')==kind],refined))
             for kind in sorted({t.get('kind','static_background') for t in holdout})}
    changes={cid:dict(rotation_deg=float(np.rad2deg(np.linalg.norm(solution.x[i*6:i*6+3]))),
                     translation_m=float(np.linalg.norm(refined[cid]['center']-cameras[cid]['center']))) for i,cid in enumerate(variable)}
    improved=after['overall']['median_px']<.90*before['overall']['median_px']
    per_camera_ok=all(after['per_camera'][c]['median_px']<=max(1.1*before['per_camera'][c]['median_px'],before['per_camera'][c]['median_px']+.3) for c in order)
    bounds_hit=bool(np.any(np.abs(solution.x[:offset])>=bounds_hi[:offset]*.995))
    kinds_ok=all(no_median_regression(v['before']['overall']['median_px'],v['after']['overall']['median_px']) for v in by_kind.values())
    accepted=bool(solution.success and improved and per_camera_ok and kinds_ok and not bounds_hit and np.isfinite(solution.x).all() and after['invalid_tracks']==0)
    report=dict(accepted=accepted,before=before,after=after,holdout_by_kind=by_kind,changes=changes,training_tracks=len(training),holdout_tracks=len(holdout),
        original_calibration_sha256=data['calibration_sha256'],tracks_sha256=sha256(args.tracks),
        solver=dict(success=bool(solution.success),message=solution.message,nfev=solution.nfev,optimality=float(solution.optimality),bounds_hit=bounds_hit),
        policy='Solver converged; >=10% heldout median improvement; no camera median regression > max(10%, .3px); no modality median regression (1e-8 px numerical tolerance); no bounds hit or invalid holdout triangulation',
        checks=dict(median_improved=bool(improved),per_camera_ok=bool(per_camera_ok),modalities_ok=bool(kinds_ok)),
        fixed_intrinsics=True,anchor_camera=anchor,metric_baseline_cameras=[anchor,'5'],
        fixed_cameras=[c for c in order if c not in variable],solver_method=args.solver,
        fit_kind=args.fit_kind,rotation_only=args.rotation_only,
        caveat='Relative correction, original anchor/world frame/metric baseline retained. Same-scene validation reused for candidate selection, not a blind test or measured 3D truth. Human associations were seeded with the original calibration.',
        elapsed_seconds=time.monotonic()-started)
    write_json(args.output/'report.json',report)
    new=deepcopy(document)
    for record in new['cameras']:
        cid=str(int(record['camera_id'].split('/')[-1]))
        if cid not in variable:continue
        matrix=np.eye(4);matrix[:3]=refined[cid]['e']
        record['world_to_camera']=matrix.tolist();record['camera_to_world']=np.linalg.inv(matrix).tolist()
        record['position_m']=refined[cid]['center'].tolist()
    new.pop('point_cloud',None)
    new['recording_refinement']=dict(report,source_calibration=str(Path(data['calibration']).resolve()),
        source_metrics_are_historical=True,point_cloud_removed='Original dense cloud was not reoptimized; do not pair it with corrected poses')
    write_json(args.output/('calibration_result.json' if accepted else 'candidate_REJECTED.json'),new)
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('extract');p.add_argument('--calibration',type=Path,default=DEFAULT_CALIBRATION)
    p.add_argument('--evidence',type=Path,default=DEFAULT_EVIDENCE);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('refine');p.add_argument('--tracks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=250)
    p.add_argument('--solver',choices=['bundle','epipolar'],default='bundle')
    p.add_argument('--refine-cameras',nargs='+',default=['2','3','4','5','6','7','8'])
    p.add_argument('--fit-kind',choices=['all','human'],default='all')
    p.add_argument('--rotation-only',action='store_true')
    p=sub.add_parser('augment');p.add_argument('--tracks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',type=Path,default=DEFAULT_LABELS)
    args=parser.parse_args();globals()[args.command](args)


if __name__=='__main__':main()
