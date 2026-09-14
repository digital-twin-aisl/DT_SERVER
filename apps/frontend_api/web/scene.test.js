import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {PeopleRenderer, parsePerson, LIMBS} from './scene.js';
const person = {global_id: 7, root: {position: [1000, 2000, 15000]}, pose: null};
const identity = new THREE.Matrix4().toArray();
test('millimetres and world joints become root-relative metres', () => {
  const p = parsePerson({...person, pose: {joint_format:'voxelpose_15j_xyz', joints:Array(15).fill([1100, 2200, 15300])}});
  assert.deepEqual(p.root, [1,2,15]);
  p.joints[0].forEach((v,i) => assert.ok(Math.abs(v - [.1,.2,.3][i]) < 1e-10));
  assert.equal(LIMBS.length, 14);
});
test('invalid root rejected; invalid or unknown pose falls back', () => {
  assert.equal(parsePerson({...person,root:{position:[NaN,0,0]}}), null);
  assert.equal(parsePerson({...person,global_id:'7'}), null);
  for (const pose of [{joint_format:'other'}, {joint_format:'voxelpose_15j_xyz',joints:Array(15).fill([0,Infinity,0])}])
    assert.equal(parsePerson({...person,pose}).joints,null);
});
test('snapshot removal, pose transitions and parent registration match Isaac', () => {
  const parent = new THREE.Matrix4().makeRotationZ(Math.PI / 2); parent.setPosition(10,20,30);
  const render = new PeopleRenderer(parent.toArray());
  const world = new THREE.Group(); world.rotation.x = -Math.PI / 2; world.add(render.group);
  render.apply({schema_version:1,people:[person]});
  const state = render.people.get(7); assert.ok(state.fallback.visible);
  world.updateMatrixWorld(true);
  const position = state.group.getWorldPosition(new THREE.Vector3());
  assert.ok(position.distanceTo(new THREE.Vector3(8,45,-21)) < 1e-10);
  render.apply({schema_version:1,people:[{...person,pose:{joint_format:'voxelpose_15j_xyz',joints:Array(15).fill([1000,2000,15000])}}]});
  assert.ok(state.skeleton.visible); assert.equal(state.fallback.visible,false); assert.ok(state.bones.every(b => !b.visible));
  render.apply({schema_version:1,people:[person]}); assert.ok(state.fallback.visible); assert.equal(state.skeleton.visible,false);
  render.apply({schema_version:1,people:[]}); assert.equal(render.people.size,0); assert.equal(render.group.children.length,0);
  render.dispose();
});
