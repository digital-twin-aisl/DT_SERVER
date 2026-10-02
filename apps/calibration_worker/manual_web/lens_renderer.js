// SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
// SPDX-License-Identifier: LGPL-2.1-or-later
import * as THREE from 'three';
import {setCalibratedCamera} from './geometry.js';

// One authoritative ray lookup is shared by the map, video overlay and picking.
export function lookupRay(model, u, v) {
  if(!model || u<0 || u>1 || v<0 || v>1)return null;
  const {width,height}=model.meta;
  const x=Math.max(0,Math.min(width-1,u*width-.5)),y=Math.max(0,Math.min(height-1,v*height-.5));
  const x0=Math.floor(x),y0=Math.floor(y),x1=Math.min(width-1,x0+1),y1=Math.min(height-1,y0+1);
  const fx=x-x0,fy=y-y0,result=[0,0];
  for(const [ix,iy,weight] of [[x0,y0,(1-fx)*(1-fy)],[x1,y0,fx*(1-fy)],[x0,y1,(1-fx)*fy],[x1,y1,fx*fy]]){
    if(!weight)continue;
    const i=(iy*width+ix)*2,a=model.rays[i],b=model.rays[i+1];
    if(!Number.isFinite(a)||!Number.isFinite(b)||Math.abs(a)>=20||Math.abs(b)>=20)return null;
    result[0]+=weight*a;result[1]+=weight*b;
  }
  return result;
}

export class LensRenderer {
  constructor(renderer, transparent=false) {
    this.renderer=renderer;this.transparent=transparent;
    this.camera=new THREE.PerspectiveCamera();this.target=new THREE.WebGLRenderTarget(1,1,{depthBuffer:true});
    this.postScene=new THREE.Scene();this.postCamera=new THREE.Camera();
    this.material=new THREE.ShaderMaterial({depthTest:false,depthWrite:false,
      uniforms:{rays:{value:null},source:{value:this.target.texture},bounds:{value:new THREE.Vector4()},empty:{value:new THREE.Vector4(.04,.06,.08,transparent?0:1)}},
      vertexShader:'varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}',
      fragmentShader:`
        uniform sampler2D rays; uniform sampler2D source;
        uniform vec4 bounds; uniform vec4 empty; varying vec2 vUv;
        void main(){
          vec2 ray=texture2D(rays,vec2(vUv.x,1.0-vUv.y)).rg;
          vec2 p=(ray-bounds.xy)/(bounds.zw-bounds.xy);
          if(any(lessThan(p,vec2(0.0)))||any(greaterThan(p,vec2(1.0)))){gl_FragColor=empty;}
          else{vec4 c=texture2D(source,vec2(p.x,1.0-p.y));gl_FragColor=c.a>0.0?c:empty;}
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`});
    this.quad=new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.material);this.quad.frustumCulled=false;this.postScene.add(this.quad);
  }
  setModel(model) {
    if(model && Math.max(model.meta.width,model.meta.height,...model.meta.source_size)>this.renderer.capabilities.maxTextureSize)throw new Error('Camera image exceeds GPU texture-size limit.');
    this.texture?.dispose();this.model=model;
    if(!model){this.material.uniforms.rays.value=null;return;}
    const m=model.meta;
    this.texture=new THREE.DataTexture(model.rays,m.width,m.height,THREE.RGFormat,THREE.FloatType);
    this.texture.minFilter=this.texture.magFilter=THREE.NearestFilter;this.texture.flipY=false;this.texture.needsUpdate=true;
    this.material.uniforms.rays.value=this.texture;this.material.uniforms.bounds.value.fromArray(m.ray_bounds);
    this.target.setSize(...m.source_size);
  }
  render(scene,pose,helpers) {
    if(!this.model){this.renderer.clear();return;}
    const m=this.model.meta;
    setCalibratedCamera(this.camera,pose,m.source_camera_matrix,...m.source_size);
    const background=scene.background,visible=helpers.visible;
    try{
      scene.background=null;helpers.visible=false;
      this.renderer.setRenderTarget(this.target);this.renderer.clear();this.renderer.render(scene,this.camera);
      this.renderer.setRenderTarget(null);this.renderer.render(this.postScene,this.postCamera);
    }finally{scene.background=background;helpers.visible=visible;this.renderer.setRenderTarget(null);}
  }
  raycaster(u,v,pose) {
    const xy=lookupRay(this.model,u,v);if(!xy)return null;
    const m=this.model.meta;setCalibratedCamera(this.camera,pose,m.source_camera_matrix,...m.source_size);
    const ray=new THREE.Raycaster();ray.ray.origin.copy(this.camera.position);
    ray.ray.direction.set(xy[0],-xy[1],-1).transformDirection(this.camera.matrixWorld);
    const distancePerDepth=Math.hypot(xy[0],xy[1],1);ray.near=.05*distancePerDepth;ray.far=2000*distancePerDepth;
    return ray;
  }
}
