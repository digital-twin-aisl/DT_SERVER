// SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
// SPDX-License-Identifier: LGPL-2.1-or-later
import * as THREE from 'three';

export const matrix = rows => new THREE.Matrix4().set(...rows.flat());
export const rows = m => Array.from({length:4}, (_,r) => Array.from({length:4}, (_,c) => m.elements[c*4+r]));
export const worldToView = p => new THREE.Vector3(p[0], p[2], -p[1]);
export const viewToWorld = p => [p.x, -p.z, p.y];

export function offsetPose(base, translation, rotation) {
  if (![...translation, ...rotation].every(Number.isFinite)) throw new Error('Offsets must be finite.');
  const m = matrix(base);
  const delta = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(...rotation.map(THREE.MathUtils.degToRad), 'ZYX'));
  m.multiply(delta);
  m.elements[12] += translation[0]; m.elements[13] += translation[1]; m.elements[14] += translation[2];
  return rows(m);
}

export function setCalibratedCamera(camera, pose, k, width, height) {
  const cvToGL = new THREE.Matrix4().makeScale(1, -1, -1);
  const worldBasis = new THREE.Matrix4().makeRotationX(-Math.PI/2);
  const m = worldBasis.multiply(matrix(pose)).multiply(cvToGL);
  m.decompose(camera.position, camera.quaternion, camera.scale);
  camera.updateMatrixWorld(true);
  const n=.05, f=2000;
  camera.projectionMatrix.set(2*k[0][0]/width, -2*k[0][1]/width, 1-2*k[0][2]/width, 0,
    0, 2*k[1][1]/height, 2*k[1][2]/height-1, 0,
    0, 0, -(f+n)/(f-n), -2*f*n/(f-n), 0, 0, -1, 0);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}
