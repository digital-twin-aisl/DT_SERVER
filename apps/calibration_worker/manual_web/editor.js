// SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
// SPDX-License-Identifier: LGPL-2.1-or-later
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { worldToView, viewToWorld, offsetPose } from './geometry.js';
import {LensRenderer} from './lens_renderer.js';
import {pixelFromPointer,renderPixelGuides} from './pixel_guide.js';
import {cloudGeometry,pickedWorldPoint} from './point_cloud.js';

const $ = id => document.getElementById(id);
const clone = value => structuredClone(value);
const error = e => { $('error-message').textContent=e.message || String(e); $('error').hidden=false; };
$('dismiss-error').onclick=()=>{$('error').hidden=true;};
const status = text => { $('status').textContent=text; };
const fmt = v => v == null ? '—' : v.toFixed(2);
async function api(path, body) {
  const response = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
function download(name, value) {
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));
  const a=document.createElement('a'); a.href=url; a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
const stamp = () => new Date().toISOString().replace(/[:.]/g,'-');

let config, state, editBase, selected=-1, projection=null, fitProposal=null, pendingPick=false;
let projectVersion=0, frameVersion=0, selectionVersion=0, projectTimer, seekTimer, frameAbort, hasUnsaved=false, frameURL;
const undo=[];
const offsets=[0,0,0,0,0,0];
let scene, renderer, overlayRenderer, orbitCamera, controls, mapModel, helpers, matchedMap, matchedOverlay;
let lensModel=null,lensVersion=0,lensAbort;
let cloudData=null,cloudObject=null,usdObject=null,referenceVersion=0;
let guidePixels=[];
const matched=()=>$('map-view').value==='camera';
let renderQueued=false;
const cameraRecord=()=>config.cameras.find(c=>c.camera_id===state.camera);
const pose=()=>state.poses[state.camera];
const pointKey=()=>`${state.dataset}:${state.mode}:${state.camera}`;
const pairs=()=>state.pairs[pointKey()] ||= [];
const requestBody=()=>({camera_id:state.camera,dataset:state.dataset,mode:state.mode,camera_to_world:pose(),pairs:pairs()});

function remember() { undo.push(clone(state)); if(undo.length>60) undo.shift(); hasUnsaved=true; }
function rebase() {
  editBase=clone(pose()); offsets.fill(0);
  for(let i=0;i<6;i++){ $(`offset-${i}`).value='0'; $(`range-${i}`).value='0'; }
}
function discardFit() {fitProposal=null;$('fit-preview').hidden=true;}
function invalidate() {
  if(!renderer || renderQueued) return;
  renderQueued=true;
  requestAnimationFrame(()=>{
    renderQueued=false;
    if(matched())matchedMap.render(scene,pose(),helpers);else renderer.render(scene,orbitCamera);
    if($('show-overlay').checked && lensModel && mapModel)matchedOverlay.render(scene,pose(),helpers);
  });
}
function rebuildHelpers() {
  if(!helpers) return;
  for(const child of [...helpers.children]) { helpers.remove(child); child.traverse(o=>{o.geometry?.dispose(); if(o.material) o.material.dispose();}); }
  for(const c of config.cameras){
    const p=state.poses[c.camera_id], origin=worldToView(p.slice(0,3).map(r=>r[3]));
    const direction=worldToView(p.slice(0,3).map(r=>r[2])).normalize();
    const color=c.camera_id===state.camera?0x5ce3d4:0xd6a45c;
    const arrow=new THREE.ArrowHelper(direction,origin,2,color,.45,.2); helpers.add(arrow);
  }
  pairs().forEach((p,i)=>{
    const marker=new THREE.Mesh(new THREE.SphereGeometry(i === selected ? .12 : .075,8,6),new THREE.MeshBasicMaterial({color:i===selected?0xffffff:0x5ce3d4,depthTest:false}));
    marker.position.copy(worldToView(p.world));marker.renderOrder=10;helpers.add(marker);
  });
  invalidate();
}
function focusCamera() {
  if(!controls) return;
  const p=pose(), c=worldToView(p.slice(0,3).map(r=>r[3]));
  const forward=worldToView(p.slice(0,3).map(r=>r[2]));
  controls.target.copy(c).addScaledVector(forward,8);
  orbitCamera.position.copy(c).add(new THREE.Vector3(12,14,12)); controls.update();invalidate();
}
function resizeMap() {
  if(!renderer) return;
  const box=$('map').getBoundingClientRect();
  const info=cameraRecord().video;
  // Backing rasters use the actual video W/H, independently of CSS size or DPR.
  renderer.setSize(matched()?info.width:box.width,matched()?info.height:box.height,false);
  orbitCamera.aspect=box.width/box.height;orbitCamera.updateProjectionMatrix();
  overlayRenderer.setSize(info.width,info.height,false);
  invalidate();
}
async function loadLensModel() {
  const version=++lensVersion;lensAbort?.abort();lensAbort=new AbortController();lensModel=null;
  matchedMap?.setModel(null);matchedOverlay?.setModel(null);$('overlay').style.display='none';
  $('lens-info').textContent='Loading the selected camera image model…';invalidate();
  try{
    const response=await fetch(`/api/lens-model?${new URLSearchParams({camera_id:state.camera,dataset:state.dataset,mode:state.mode})}`,{signal:lensAbort.signal});
    if(!response.ok)throw new Error((await response.json()).detail);
    const meta=JSON.parse(response.headers.get('X-Lens-Model')),rays=new Float32Array(await response.arrayBuffer());
    if(version!==lensVersion)return;
    if(meta.source_sha256!==state.source_sha256||rays.length!==meta.width*meta.height*2)throw new Error('Lens model does not match the loaded calibration. Reload your saved project.');
    lensModel={meta,rays};matchedMap?.setModel(lensModel);matchedOverlay?.setModel(lensModel);
    const k=meta.camera_matrix,d=meta.distortion_coefficients;
    $('lens-info').textContent=`${meta.width} × ${meta.height} · fx ${k[0][0].toFixed(2)}, fy ${k[1][1].toFixed(2)} px · cx ${k[0][2].toFixed(2)}, cy ${k[1][2].toFixed(2)} · D [${d.map(v=>v.toFixed(5)).join(', ')}]${meta.invalid_fraction>0?` · ${(100*meta.invalid_fraction).toFixed(2)}% non-invertible lens pixels masked`:''}`;
    $('overlay').style.display=$('show-overlay').checked?'block':'none';invalidate();
  }catch(e){if(version===lensVersion&&e.name!=='AbortError'){lensModel=null;matchedMap?.setModel(null);matchedOverlay?.setModel(null);$('overlay').style.display='none';invalidate();error(e);$('lens-info').textContent='Camera model unavailable. Matched rendering disabled; no pinhole fallback.';}}
}
function updateMapMode() {
  if(controls)controls.enabled=!matched()&&!pendingPick;
  $('focus').disabled=matched();
  $('map-hint').textContent=matched()?'Selected video intrinsics, lens distortion and exact W/H; current edited pose. Pick landmark uses the same inverse lens model.':'Free orbit for landmark selection only — not an image-matched comparison. Drag to orbit, right-drag to pan, scroll to zoom.';
  resizeMap();
  drawPixelGuide();
}
function drawPixelGuide() {
  const {width,height}=cameraRecord().video;
  const remove=index=>{guidePixels.splice(index,1);drawPixelGuide();status('Guide dot removed from both views. Landmarks unchanged.');};
  renderPixelGuides($('video-pixel-guide'),guidePixels,width,height,remove);
  renderPixelGuides($('map-pixel-guide'),guidePixels,width,height,remove,matched());
  $('clear-pixel-guide').disabled=!guidePixels.length;
  $('pixel-guide-info').textContent=guidePixels.length
    ? `${guidePixels.length} guide dots · Click a dot to remove${matched()?'':' · Map guides hidden in Free orbit'}`
    : 'Click the video to add red guide dots in both views.';
}
function clearPixelGuide() {guidePixels=[];drawPixelGuide();}
function updateCloud() {
  if(!cloudData)return;
  const confidence=$('cloud-confidence').valueAsNumber,size=$('cloud-size').valueAsNumber;
  if(!Number.isFinite(confidence)||!Number.isFinite(size)||size<=0)return;
  const exclude=$('cloud-exclude').checked?config.point_cloud.sources.findIndex(s=>s.camera_id===state.camera):-1;
  const geometry=cloudGeometry(cloudData,{minConfidence:confidence,excludeSource:exclude});
  if(!cloudObject){
    cloudObject=new THREE.Points(geometry,new THREE.PointsMaterial({vertexColors:true,size,sizeAttenuation:true}));
    cloudObject.name='VGGT cloud';scene.add(cloudObject);
  }else{cloudObject.geometry.dispose();cloudObject.geometry=geometry;cloudObject.material.size=size;}
  cloudObject.visible=$('reference-source').value==='cloud';
  $('cloud-info').textContent=`${geometry.getAttribute('position').count.toLocaleString()} / ${config.point_cloud.point_count.toLocaleString()} points · ${config.point_cloud.pose_stage} poses · Dense depth not BA-refined. Not ground truth.`;
  invalidate();
}
async function loadReference() {
  const version=++referenceVersion,isCloud=$('reference-source').value==='cloud';
  if(usdObject)usdObject.visible=false;if(cloudObject)cloudObject.visible=false;
  mapModel=null;$('map-status').hidden=false;$('map-status').textContent=isCloud?'Loading paired VGGT point cloud…':'Loading local USD map…';
  $('cloud-controls').hidden=!isCloud;invalidate();
  try{
    if(isCloud){
      if(!cloudData){
        const response=await fetch('/api/point-cloud');
        if(!response.ok)throw new Error((await response.json()).detail);
        if(response.headers.get('X-Calibration-SHA256')!==state.source_sha256)throw new Error('Point cloud belongs to another calibration. Reload the editor.');
        const data=new Float32Array(await response.arrayBuffer());
        if(data.length!==config.point_cloud.point_count*8)throw new Error('Point-cloud size mismatch.');
        cloudData=data;
      }
      if(version!==referenceVersion)return;updateCloud();mapModel=cloudObject;
    }else{
      if(!usdObject){
        const loader=new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
        const gltf=await loader.loadAsync('/assets/map.glb');
        // Reuse the first load even when a selection changes in flight.
        if(!usdObject){usdObject=gltf.scene;usdObject.visible=false;scene.add(usdObject);}
      }
      if(version!==referenceVersion)return;mapModel=usdObject;
    }
    mapModel.visible=true;$('map-status').hidden=true;invalidate();
  }catch(e){if(version===referenceVersion){$('map-status').textContent='Reference failed to load. Check the paired calibration/cloud files.';error(e);}}
}
function setupMap() {
  scene=new THREE.Scene(); scene.background=new THREE.Color('#24343e');
  scene.add(new THREE.HemisphereLight(0xe7f4ff,0x758575,2.5));
  const sun=new THREE.DirectionalLight(0xfff3df,3);sun.position.set(150,250,100);scene.add(sun);
  renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(1);
  renderer.toneMapping=THREE.ACESFilmicToneMapping;$('map').append(renderer.domElement);
  overlayRenderer=new THREE.WebGLRenderer({alpha:true,antialias:true});overlayRenderer.setPixelRatio(1);
  overlayRenderer.toneMapping=THREE.ACESFilmicToneMapping;overlayRenderer.setClearColor(0,0);$('overlay').append(overlayRenderer.domElement);
  orbitCamera=new THREE.PerspectiveCamera(55,1,.1,3000);
  matchedMap=new LensRenderer(renderer);matchedOverlay=new LensRenderer(overlayRenderer,true);
  controls=new OrbitControls(orbitCamera,renderer.domElement);controls.addEventListener('change',invalidate);controls.minDistance=.3;controls.maxDistance=1500;
  helpers=new THREE.Group();scene.add(helpers);
  loadReference();
  renderer.domElement.addEventListener('click',event=>{
    if(!pendingPick||!mapModel)return;
    const box=renderer.domElement.getBoundingClientRect();
    const u=(event.clientX-box.left)/box.width,v=(event.clientY-box.top)/box.height;
    let ray;
    if(matched()){
      ray=matchedMap.raycaster(u,v,pose());
      if(!ray){status('This pixel has no valid inverse lens ray. Pick another point or use Free orbit.');return;}
    }else{ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2(u*2-1,1-v*2),orbitCamera);}
    ray.params.Points.threshold=Math.max(.01,$('cloud-size').valueAsNumber||.03);
    mapModel.updateMatrixWorld(true);
    const hit=ray.intersectObject(mapModel,true)[0];
    if(!hit){status('No reference hit. Pick a visible structure or increase the point size.');return;}
    addPoint(viewToWorld(pickedWorldPoint(hit)));pendingPick=false;updateMapMode();$('pick').classList.remove('active');
  });
  new ResizeObserver(resizeMap).observe($('map'));new ResizeObserver(resizeMap).observe($('image-stage'));
  rebuildHelpers();focusCamera();resizeMap();
}

function drawMarks() {
  const canvas=$('marks'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  const scale=canvas.width/Math.max(canvas.clientWidth,1);ctx.font=`${13*scale}px system-ui`;ctx.lineWidth=1.5*scale;
  function circle(xy,color,index,dashed=false){
    if(!xy)return;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.setLineDash(dashed?[4*scale,3*scale]:[]);
    ctx.beginPath();ctx.arc(xy[0],xy[1],(index===selected?6:4)*scale,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);
    ctx.fillText(String(index+1),xy[0]+8*scale,xy[1]-6*scale);
  }
  pairs().forEach((p,i)=>{
    const xy=projection?.current.points[i]?.projected;
    if($('show-original').checked)circle(projection?.original.points[i]?.projected,'#ffc86f',i,true);
    circle(xy,'#5ce3d4',i);
    if(p.image){
      const [x,y]=p.image;ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(x-6*scale,y);ctx.lineTo(x+6*scale,y);ctx.moveTo(x,y-6*scale);ctx.lineTo(x,y+6*scale);ctx.stroke();
      if(xy){ctx.strokeStyle='#ff7285';ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(...xy);ctx.stroke();}
    }
  });
}
function renderPoints() {
  $('point-count').textContent=String(pairs().length);$('empty-points').hidden=!!pairs().length;$('points').replaceChildren();
  pairs().forEach((p,i)=>{
    const tr=document.createElement('tr');if(i===selected)tr.className='selected';
    const choose=document.createElement('button');choose.textContent=`#${i+1}`;choose.setAttribute('aria-label',`Select landmark ${i+1}`);
    const first=document.createElement('td');first.append(choose);tr.append(first);
    for(const text of [p.world.map(x=>x.toFixed(3)).join(', '),p.image?p.image.map(x=>x.toFixed(1)).join(', '):'Click in video',fmt(projection?.original.points[i]?.error_px),fmt(projection?.current.points[i]?.error_px)]){
      const td=document.createElement('td');td.textContent=text;tr.append(td);
    }
    const td=document.createElement('td'),check=document.createElement('input');check.type='checkbox';check.checked=p.holdout;check.setAttribute('aria-label',`Use point ${i+1} as holdout`);
    check.onclick=e=>e.stopPropagation();check.onchange=()=>{remember();p.holdout=check.checked;changed();};td.append(check);tr.append(td);
    tr.onclick=()=>{selected=i;renderPoints();drawMarks();rebuildHelpers();status(`Point ${i+1} selected. Click its observed position in the video.`);};$('points').append(tr);
  });
  $('delete-point').disabled=selected<0;
}
function renderReadouts() {
  const changed=JSON.stringify(pose())!==JSON.stringify(cameraRecord().camera_to_world);
  $('dirty').textContent=changed?'Edited pose':'Original pose';
  $('position').textContent=pose().slice(0,3).map(r=>r[3].toFixed(4)).join(' / ');
  $('before-error').textContent=fmt(projection?.original.metrics.all.rmse_px);
  $('after-error').textContent=fmt(projection?.current.metrics.all.rmse_px);
  $('holdout-error').textContent=fmt(projection?.current.metrics.holdout.rmse_px);
  const observed=pairs().filter(p=>p.image).length,held=pairs().filter(p=>p.image&&p.holdout).length;
  $('before-count').textContent=`${projection?.original.metrics.all.count??0}/${observed} points`;
  $('after-count').textContent=`${projection?.current.metrics.all.count??0}/${observed} points`;
  $('holdout-count').textContent=`${projection?.current.metrics.holdout.count??0}/${held} points`;
  $('undo').disabled=!undo.length;
}
async function projectNow() {
  const version=++projectVersion;
  try{
    const result=await api('/api/project',requestBody());
    if(version!==projectVersion)return;
    projection=result;renderPoints();renderReadouts();drawMarks();invalidate();
  }catch(e){if(version===projectVersion)error(e);}
}
function changed() {
  discardFit();projection=null;projectVersion++;clearTimeout(projectTimer);
  renderReadouts();renderPoints();drawMarks();rebuildHelpers();
  projectTimer=setTimeout(projectNow,100);
}
function addPoint(world) {
  if(world.length!==3||!world.every(Number.isFinite))throw new Error('Enter finite XYZ coordinates in metres.');
  if(pairs().length>=200)throw new Error('Limit: 200 landmarks per camera / dataset / image model.');
  remember();pairs().push({world,image:null,holdout:false});selected=pairs().length-1;changed();status(`Point ${selected+1} added. Click its matching location in the video.`);
}
async function loadFrame() {
  clearPixelGuide();
  const version=++frameVersion,record=cameraRecord().video;
  frameAbort?.abort();frameAbort=new AbortController();
  const index=Math.max(0,Math.min(record.frames-1,Math.round(state.time*record.fps)));
  state.time=index/record.fps;$('timeline').max=record.frames-1;$('timeline').value=index;$('seconds').max=record.duration;$('seconds').value=state.time.toFixed(3);
  $('frame-info').textContent=`${index} / ${record.frames-1} · ${state.time.toFixed(3)} s`;
  $('image-stage').style.aspectRatio=`${record.width}/${record.height}`;$('marks').width=record.width;$('marks').height=record.height;
  $('marks').style.pointerEvents='none';$('frame').style.opacity='.4';
  try{
    const url=`/api/frame?${new URLSearchParams({camera_id:state.camera,dataset:state.dataset,index,mode:state.mode})}`;
    const response=await fetch(url,{signal:frameAbort.signal});if(!response.ok)throw new Error((await response.json()).detail);
    const object=URL.createObjectURL(await response.blob());
    const image=new Image();image.src=object;await image.decode();
    if(version!==frameVersion){URL.revokeObjectURL(object);return;}
    $('frame').src=object;if(frameURL)URL.revokeObjectURL(frameURL);frameURL=object;
    $('marks').style.pointerEvents='auto';$('frame').style.opacity='1';drawMarks();
  }catch(e){if(version===frameVersion&&e.name!=='AbortError')error(e);}
}
async function activate() {
  clearTimeout(projectTimer);clearTimeout(seekTimer);pendingPick=false;$('pick').classList.remove('active');
  selected=-1;projection=null;discardFit();projectVersion++;rebase();
  $('camera').value=state.camera;$('dataset').value=state.dataset;$('mode').value=state.mode;
  $('show-overlay').disabled=false;
  const info=cameraRecord().video;$('map').style.aspectRatio=`${info.width}/${info.height}`;
  if(cloudData)updateCloud();
  updateMapMode();renderPoints();renderReadouts();rebuildHelpers();focusCamera();await Promise.all([loadFrame(),projectNow(),loadLensModel()]);
}

function buildControls() {
  for(let i=0;i<6;i++){
    const div=document.createElement('div');div.className='axis';
    const label=document.createElement('label');label.htmlFor=`offset-${i}`;label.textContent='XYZ'[i%3];div.append(label);
    const minus=document.createElement('button');minus.textContent='−';minus.setAttribute('aria-label',`Decrease ${i<3?'position':'rotation'} ${label.textContent}`);div.append(minus);
    const number=document.createElement('input');number.id=`offset-${i}`;number.type='number';number.value=0;number.step=i<3?.05:.25;number.setAttribute('aria-label',`${i<3?'Position':'Rotation'} ${label.textContent} offset`);div.append(number);
    const plus=document.createElement('button');plus.textContent='+';plus.setAttribute('aria-label',`Increase ${i<3?'position':'rotation'} ${label.textContent}`);div.append(plus);
    const range=document.createElement('input');range.type='range';range.id=`range-${i}`;range.min=i<3?-2:-15;range.max=i<3?2:15;range.step=number.step;range.value=0;range.setAttribute('aria-label',`${i<3?'Position':'Rotation'} ${label.textContent} offset`);div.append(range);
    function apply(value){if(!Number.isFinite(value))return;offsets[i]=value;number.value=value;range.value=value;state.poses[state.camera]=offsetPose(editBase,offsets.slice(0,3),offsets.slice(3));changed();}
    number.onfocus=remember;range.onpointerdown=remember;range.onkeydown=e=>{if(e.key.startsWith('Arrow'))remember();};
    number.oninput=()=>apply(number.valueAsNumber);range.oninput=()=>apply(range.valueAsNumber);
    function nudge(sign){const step=$(i<3?'move-step':'rotate-step').valueAsNumber;if(!Number.isFinite(step)||step<=0)return;remember();apply(Number((offsets[i]+sign*step).toFixed(8)));}
    minus.onclick=()=>nudge(-1);plus.onclick=()=>nudge(1);
    $(i<3?'translation-controls':'rotation-controls').append(div);
  }
}
async function exportCalibration() {
  const edits={};for(const c of config.cameras)if(JSON.stringify(state.poses[c.camera_id])!==JSON.stringify(c.camera_to_world))edits[c.camera_id]=state.poses[c.camera_id];
  const document=await api('/api/export',{source_sha256:state.source_sha256,edits});
  download(`calibration_manual_${stamp()}.json`,document);status('Calibration downloaded. No deployment files were changed. Save the project too to preserve landmarks.');
  return {edited_camera_ids:Object.keys(edits)};
}

async function main() {
  config=await api('/api/config?dataset=1');
  state={schema_version:1,source_sha256:config.source_sha256,dataset:1,camera:config.cameras[0].camera_id,mode:'raw',time:10,poses:Object.fromEntries(config.cameras.map(c=>[c.camera_id,clone(c.camera_to_world)])),pairs:{}};
  for(const c of config.cameras){const option=document.createElement('option');option.value=c.camera_id;option.textContent=c.camera_id;$('camera').append(option);}
  $('source').textContent=`Source: ${config.source_name}. Intrinsic reference size: ${config.calibration_size_assumed.join(' × ')} px (unless recorded in camera metadata).`;
  $('reference-source').value=config.point_cloud?'cloud':'usd';
  $('reference-cloud-option').disabled=!config.point_cloud;
  if(config.point_cloud){$('cloud-confidence').value=config.point_cloud.min_confidence;$('cloud-confidence').min=config.point_cloud.min_confidence;}
  buildControls();rebase();try{setupMap();}catch(e){error(e);$('map-status').textContent='WebGL unavailable. Use world-coordinate entry.';}
  $('reference-source').onchange=()=>{pendingPick=false;$('pick').classList.remove('active');updateMapMode();loadReference();};
  $('cloud-confidence').onchange=updateCloud;$('cloud-size').onchange=updateCloud;$('cloud-exclude').onchange=updateCloud;
  $('camera').onchange=async()=>{state.camera=$('camera').value;await activate();};
  $('dataset').onchange=async()=>{const version=++selectionVersion;try{const dataset=Number($('dataset').value);const next=await api(`/api/config?dataset=${dataset}`);if(version!==selectionVersion)return;state.dataset=dataset;config=next;await activate();}catch(e){if(version===selectionVersion){$('dataset').value=state.dataset;error(e);}}};
  $('mode').onchange=async()=>{state.mode=$('mode').value;await activate();status('Image-model-specific landmarks loaded. Points are not reused across image models.');};
  $('timeline').oninput=()=>{state.time=Number($('timeline').value)/cameraRecord().video.fps;clearTimeout(seekTimer);seekTimer=setTimeout(loadFrame,100);};
  $('seconds').onchange=()=>{if(Number.isFinite($('seconds').valueAsNumber)){state.time=$('seconds').valueAsNumber;loadFrame();}};
  $('prev-frame').onclick=()=>{state.time-=1/cameraRecord().video.fps;loadFrame();};$('next-frame').onclick=()=>{state.time+=1/cameraRecord().video.fps;loadFrame();};
  $('marks').onclick=e=>{
    const pixel=pixelFromPointer(e,$('marks').getBoundingClientRect(),$('marks').width,$('marks').height);
    if(!pixel)return;
    guidePixels.push(pixel);drawPixelGuide();
    if($('guide-only').checked||selected<0){status('Pixel guide placed. No landmark or camera pose changed.');return;}
    remember();pairs()[selected].image=pixel;changed();
  };
  $('clear-pixel-guide').onclick=clearPixelGuide;
  $('pick').onclick=()=>{if(!mapModel){status('Wait for the map to load, or enter coordinates.');return;}pendingPick=!pendingPick;updateMapMode();$('pick').classList.toggle('active',pendingPick);status(pendingPick?'Click a static surface on the 3D map.':'Landmark picking cancelled.');};
  $('map-view').onchange=updateMapMode;
  $('focus').onclick=focusCamera;$('show-original').onchange=drawMarks;
  $('show-overlay').onchange=()=>{$('overlay').style.display=$('show-overlay').checked&&lensModel?'block':'none';invalidate();};
  $('opacity').oninput=()=>{$('overlay').style.opacity=$('opacity').value;};
  for(const [id,start] of [['move-step',0],['rotate-step',3]])$(id).onchange=()=>{const step=$(id).valueAsNumber;if(Number.isFinite(step)&&step>0)for(let i=start;i<start+3;i++){$(`offset-${i}`).step=step;$(`range-${i}`).step=step;}};
  $('add-point').onclick=()=>{try{addPoint(['x','y','z'].map(a=>$(`point-${a}`).valueAsNumber));}catch(e){error(e);}};
  $('delete-point').onclick=()=>{if(selected>=0){remember();pairs().splice(selected,1);selected=-1;changed();}};
  $('rebase').onclick=()=>{rebase();status('Controls zeroed at current edited pose; calibration is unchanged.');};
  $('reset').onclick=()=>{remember();state.poses[state.camera]=clone(cameraRecord().camera_to_world);rebase();changed();};
  $('undo').onclick=async()=>{if(!undo.length)return;const previous=undo.pop();if(previous.dataset!==state.dataset)config=await api(`/api/config?dataset=${previous.dataset}`);state=previous;hasUnsaved=true;await activate();};
  $('refine').onclick=async()=>{
    const snapshot=JSON.stringify(requestBody());$('refine').disabled=true;
    try{
      const result=await api('/api/refine',requestBody());
      if(JSON.stringify(requestBody())!==snapshot){status('Inputs changed. Discarded stale refinement proposal.');return;}
      fitProposal=result;$('fit-preview').hidden=false;
      $('fit-summary').textContent=`Proposed fit RMSE: ${fmt(result.before.metrics.fit.rmse_px)} → ${fmt(result.after.metrics.fit.rmse_px)} px. Holdout: ${fmt(result.before.metrics.holdout.rmse_px)} → ${fmt(result.after.metrics.holdout.rmse_px)} px. Not applied.${result.at_bound?' WARNING: adjustment limit reached.':''}${result.coplanar?' Points are coplanar; add landmarks at other heights.':''}`;
    }catch(e){error(e);}finally{$('refine').disabled=false;}
  };
  $('apply-fit').onclick=()=>{if(!fitProposal)return;remember();state.poses[state.camera]=clone(fitProposal.camera_to_world);rebase();changed();status('Proposed pose applied locally. Compare holdout points before export.');};$('cancel-fit').onclick=discardFit;
  $('save-project').onclick=()=>{download(`camera_pose_project_${stamp()}.json`,state);hasUnsaved=false;status('Project downloaded with poses and image-model-specific landmarks.');};
  $('open-project-button').onclick=()=>{$('open-project').click();};
  $('open-project').onchange=async()=>{
    try{
      const file=$('open-project').files[0];if(!file)return;if(file.size>5e6)throw new Error('Project file exceeds 5 MB.');
      const loaded=JSON.parse(await file.text());await validateProject(loaded);
      const next=await api(`/api/config?dataset=${loaded.dataset}`);
      remember();state=loaded;config=next;await activate();status('Project restored. Source calibration has not been changed.');
    }catch(e){error(e);}finally{$('open-project').value='';}
  };
  $('export').onclick=()=>exportCalibration().catch(error);
  addEventListener('beforeunload',e=>{if(hasUnsaved){e.preventDefault();e.returnValue='';}});
  await activate();status('Ready. Choose the correct image model, then mark static landmarks.');
  registerTools();
}

async function validateProject(value) {
  if(value.schema_version!==1||value.source_sha256!==config.source_sha256)throw new Error('Project source calibration does not match.');
  if(![1,2,3].includes(value.dataset)||!['raw','rectified','pinhole'].includes(value.mode)||!Number.isFinite(value.time)||value.time<0)throw new Error('Invalid project selection.');
  const ids=config.cameras.map(c=>c.camera_id);
  if(!ids.includes(value.camera)||!value.poses||Object.keys(value.poses).length!==ids.length||!ids.every(id=>value.poses[id]))throw new Error('Project must contain exactly the source cameras.');
  if(!value.pairs||typeof value.pairs!=='object'||Array.isArray(value.pairs))throw new Error('Invalid landmark collection.');
  for(const [key,points] of Object.entries(value.pairs)){
    const [dataset,mode,camera]=key.split(':');
    if(![1,2,3].includes(Number(dataset))||!['raw','rectified','pinhole'].includes(mode)||!ids.includes(camera)||!Array.isArray(points)||points.length>200)throw new Error('Invalid landmark group.');
    await api('/api/project',{camera_id:camera,dataset:Number(dataset),mode,camera_to_world:value.poses[camera],pairs:points});
  }
  await api('/api/export',{source_sha256:value.source_sha256,edits:value.poses});
}

function registerTools() {
  const context=document.modelContext;if(!context?.registerTool)return;
  const lifecycle=new AbortController();addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
  for(const tool of [
    {name:'read_camera_pose',description:'Read the selected camera pose and reprojection errors; no changes.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute:async input=>{if(!input||Array.isArray(input)||Object.keys(input).length)throw new Error('Expected an empty object.');return {camera_id:state.camera,camera_to_world:clone(pose()),metrics:projection?.current.metrics??null};}},
    {name:'stage_camera_offsets',description:'Stage world XYZ translation in metres and camera-local XYZ rotation in degrees, relative to the currently edited camera pose. Does not export or deploy.',inputSchema:{type:'object',properties:{translation:{type:'array',items:{type:'number'},minItems:3,maxItems:3},rotation:{type:'array',items:{type:'number'},minItems:3,maxItems:3}},required:['translation','rotation'],additionalProperties:false},annotations:{readOnlyHint:false},execute:async input=>{
      if(!input||Object.keys(input).some(k=>!['translation','rotation'].includes(k))||!['translation','rotation'].every(k=>Array.isArray(input[k])&&input[k].length===3&&input[k].every(Number.isFinite)))throw new Error('Expected finite XYZ translation and rotation arrays.');
      const next=offsetPose(pose(),input.translation,input.rotation);await api('/api/project',{...requestBody(),camera_to_world:next});
      remember();state.poses[state.camera]=next;rebase();changed();clearTimeout(projectTimer);await projectNow();return {camera_id:state.camera,camera_to_world:clone(pose()),deployed:false};
    }}
  ]){try{Promise.resolve(context.registerTool(tool,{signal:lifecycle.signal})).catch(error);}catch(e){error(e);}}
}

main().catch(error);
