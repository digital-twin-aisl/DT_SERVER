// SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
// SPDX-License-Identifier: LGPL-2.1-or-later
import * as THREE from 'three';

export function cloudGeometry(data, {minConfidence=0, excludeSource=-1}={}) {
  if(data.length%8)throw new Error('Invalid cloud buffer length.');
  const positions=[],colors=[],color=new THREE.Color();
  for(let i=0;i<data.length;i+=8){
    if(data[i+6]<minConfidence||data[i+7]===excludeSource)continue;
    // Persisted cloud is USD Z-up metres, like camera poses and landmarks.
    positions.push(data[i],data[i+2],-data[i+1]);
    color.setRGB(data[i+3],data[i+4],data[i+5],THREE.SRGBColorSpace);
    colors.push(color.r,color.g,color.b);
  }
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
  geometry.computeBoundingSphere();return geometry;
}

export function pickedWorldPoint(hit) {
  // Three Points.raycast reports the point on the ray, not the source vertex.
  if(hit.object.isPoints){
    return new THREE.Vector3().fromBufferAttribute(hit.object.geometry.getAttribute('position'),hit.index)
      .applyMatrix4(hit.object.matrixWorld);
  }
  return hit.point;
}
