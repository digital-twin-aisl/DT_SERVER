// SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
// SPDX-License-Identifier: LGPL-2.1-or-later
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {matrix,rows,worldToView,viewToWorld,offsetPose,setCalibratedCamera} from './geometry.js';
import {lookupRay,LensRenderer} from './lens_renderer.js';
import {pixelFromPointer,placePixelGuide,renderPixelGuides} from './pixel_guide.js';
import {cloudGeometry,pickedWorldPoint} from './point_cloud.js';
const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
test('point cloud preserves world coordinates and filters confidence/source independently',()=>{
  const data=new Float32Array([85,5,14,1,0,0,3,0, 86,6,15,0,1,0,2,1, 87,7,16,0,0,1,.5,1]);
  const geometry=cloudGeometry(data,{minConfidence:1,excludeSource:0});
  assert.deepEqual(Array.from(geometry.attributes.position.array),[86,15,-6]);
  assert.deepEqual(Array.from(geometry.attributes.color.array),[0,1,0]);
  geometry.dispose();
});
test('cloud landmark picking uses the actual vertex, not the ray approximation',()=>{
  const geometry=cloudGeometry(new Float32Array([85,5,14,1,1,1,3,0]));
  const object=new THREE.Points(geometry);object.position.x=2;object.updateMatrixWorld();
  assert.deepEqual(pickedWorldPoint({object,index:0,point:new THREE.Vector3()}).toArray(),[87,14,-5]);
  geometry.dispose();object.material.dispose();
});
function guideContainer() {
  return {children:[],ownerDocument:{createElement:()=>({style:{},setAttribute(name,value){this[name]=value;}})},
    replaceChildren(){this.children=[];},append(child){this.children.push(child);}};
}
test('multiple guides appear at identical pixels in both views and delete in sync',()=>{
  const video=guideContainer(),map=guideContainer(),pixels=[[100,50],[900,500],[1800,1000]];
  const draw=()=>{for(const container of [video,map])renderPixelGuides(container,pixels,1920,1080,index=>{pixels.splice(index,1);draw();});};
  draw();assert.equal(video.children.length,3);assert.equal(map.children.length,3);
  video.children.forEach((dot,i)=>assert.deepEqual(dot.style,map.children[i].style));
  let stopped=false;map.children[1].onclick({stopPropagation(){stopped=true;}});
  assert.equal(stopped,true);assert.deepEqual(pixels,[[100,50],[1800,1000]]);
  assert.equal(video.children.length,2);assert.equal(map.children.length,2);
  video.children[0].onclick({stopPropagation(){}});
  assert.deepEqual(pixels,[[1800,1000]]);assert.equal(map.children.length,1);
});
test('guide layers clear all dots and hide in orbit without losing the collection',()=>{
  const layer=guideContainer(),pixels=[[100,50],[900,500]];
  renderPixelGuides(layer,pixels,1920,1080,()=>{},false);
  assert.equal(layer.hidden,true);assert.equal(layer.children.length,0);assert.equal(pixels.length,2);
  renderPixelGuides(layer,pixels,1920,1080,()=>{});assert.equal(layer.children.length,2);
  renderPixelGuides(layer,[],1920,1080,()=>{});assert.equal(layer.hidden,true);assert.equal(layer.children.length,0);
});
test('pixel guide maps CSS clicks to native pixels without rounding or distortion',()=>{
  const pixel=pixelFromPointer({clientX:340,clientY:155},{left:100,top:20,width:960,height:540},1920,1080);
  assert.deepEqual(pixel,[480,270]);
  const small={style:{}},large={style:{}};
  placePixelGuide(small,pixel,1920,1080);placePixelGuide(large,pixel,1920,1080);
  assert.deepEqual(small,large);assert.deepEqual(small.style,{left:'25%',top:'25%'});assert.equal(small.hidden,false);
});
test('pixel guide rejects outside or unmeasurable clicks and preserves subpixels',()=>{
  const box={left:10,top:20,width:100,height:50};
  assert.deepEqual(pixelFromPointer({clientX:10.25,clientY:20.25},box,200,100),[.5,.5]);
  for(const [clientX,clientY] of [[9,20],[110,20],[10,70],[NaN,20]])assert.equal(pixelFromPointer({clientX,clientY},box,200,100),null);
  assert.equal(pixelFromPointer({clientX:10,clientY:20},{...box,width:0},200,100),null);
});
test('pixel guide hides in orbit or when cleared, and restores the same pixel',()=>{
  const element={style:{}};
  placePixelGuide(element,[100,50],200,100,false);assert.equal(element.hidden,true);
  placePixelGuide(element,[100,50],200,100,true);assert.equal(element.hidden,false);assert.equal(element.style.left,'50%');
  placePixelGuide(element,null,200,100);assert.equal(element.hidden,true);
});
function near(a,b,tolerance=1e-9){assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<tolerance,`${v} vs ${b[i]}`));}

test('USD Z-up and Three Y-up coordinate mapping is reversible',()=>{
  const p=[85,5,14];near(worldToView(p).toArray(),[85,14,-5]);near(viewToWorld(worldToView(p)),p);
});
test('matrix rows are not transposed when passing through Three',()=>{
  const pose=offsetPose(identity,[85,5,18],[5,10,15]);near(rows(matrix(pose)).flat(),pose.flat());
});
test('offset translations use world axes; rotation uses camera-local axes',()=>{
  const base=offsetPose(identity,[80,5,17],[0,0,90]);
  const result=offsetPose(base,[1,2,3],[90,0,0]);
  near(result.slice(0,3).map(r=>r[3]),[81,7,20]);
  near(result.slice(0,3).flatMap(r=>r.slice(0,3)),[0,0,1,1,0,0,0,1,0]);
});
test('calibrated Three projection agrees with OpenCV pinhole coordinates',()=>{
  const pose=offsetPose(identity,[85,5,18],[10,25,40]);
  const k=[[1000,0,1003],[0,900,533],[0,0,1]],w=1920,h=1080,camera=new THREE.PerspectiveCamera();
  setCalibratedCamera(camera,pose,k,w,h);
  for(const cameraPoint of [[0,0,10],[1,2,10],[-3,-2,8]]){
    const world=new THREE.Vector3(...cameraPoint).applyMatrix4(matrix(pose));
    const ndc=worldToView(world.toArray()).project(camera);
    near([(ndc.x+1)*w/2,(1-ndc.y)*h/2],[k[0][0]*cameraPoint[0]/cameraPoint[2]+k[0][2],k[1][1]*cameraPoint[1]/cameraPoint[2]+k[1][2]],1e-8);
  }
});
test('nonfinite offsets are rejected',()=>assert.throws(()=>offsetPose(identity,[NaN,0,0],[0,0,0])));

const rayModel=()=>({meta:{width:2,height:2,source_camera_matrix:[[1,0,1],[0,1,1],[0,0,1]],source_size:[2,2]},
  rays:new Float32Array([-.5,-.5,.5,-.5,-.5,.5,.5,.5])});
test('ray lookup handles top-down rows and bilinear subpixel picking',()=>{
  near(lookupRay(rayModel(),.25,.25),[-.5,-.5]);near(lookupRay(rayModel(),.5,.5),[0,0]);
  near(lookupRay(rayModel(),.625,.625),[.25,.25]);assert.equal(lookupRay(rayModel(),-1,.5),null);
});
test('masked lens pixels cannot be used as landmark rays',()=>{
  const model=rayModel();model.rays[0]=1e9;assert.equal(lookupRay(model,.25,.25),null);
  assert.equal(lookupRay(model,.5,.5),null);near(lookupRay(model,.75,.75),[.5,.5]);
});
test('matched picking uses the same camera-local ray and edited world pose',()=>{
  const pose=offsetPose(identity,[85,5,18],[10,25,40]);
  const ray=LensRenderer.prototype.raycaster.call({model:rayModel(),camera:new THREE.PerspectiveCamera()},.75,.25,pose);
  const cameraPoint=new THREE.Vector3(.5,-.5,1).applyMatrix4(matrix(pose));
  const origin=worldToView([85,5,18]),direction=worldToView(cameraPoint.toArray()).sub(origin).normalize();
  near(ray.ray.origin.toArray(),origin.toArray());near(ray.ray.direction.toArray(),direction.toArray());
});
