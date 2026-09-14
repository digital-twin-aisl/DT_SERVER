import * as THREE from 'three';

export const LIMBS = [[0,1],[0,2],[0,3],[3,4],[4,5],[0,9],[9,10],[10,11],[2,6],[2,12],[6,7],[7,8],[12,13],[13,14]];
const PALETTE = [0xe61a4a, 0x3db54a, 0xffe01a, 0x0082c7, 0xf58230, 0x911fb5];
const point = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);

export function parsePerson(person) {
  if (!person || !Number.isSafeInteger(person.global_id) || !point(person.root?.position)) return null;
  const root = person.root.position.map(v => v / 1000);
  const pose = person.pose;
  const joints = pose?.joint_format === 'voxelpose_15j_xyz' && Array.isArray(pose.joints) &&
    pose.joints.length === 15 && pose.joints.every(point)
    ? pose.joints.map(j => j.map((v, i) => v / 1000 - root[i])) : null;
  return {id: person.global_id, root, joints, lod: person.lod, edge: person.root.edge_id};
}

export class PeopleRenderer {
  constructor(matrix) {
    // Same as Isaac: root/joints are millimetres in MetaSejong_People space.
    this.group = new THREE.Group();
    this.group.matrix.fromArray(matrix); this.group.matrixAutoUpdate = false;
    this.people = new Map();
    this.xray = true;
    this.sphere = new THREE.SphereGeometry(.04, 8, 6);
    this.bone = new THREE.CylinderGeometry(.018, .018, 1, 6);
    this.capsule = new THREE.CapsuleGeometry(.3, 1.7, 4, 8);
    this.capsule.rotateX(Math.PI / 2);
    this.axis = new THREE.Vector3(0, 1, 0);
  }
  create(id) {
    const material = new THREE.MeshStandardMaterial({color: PALETTE[((id % 6) + 6) % 6], roughness: .55, depthTest: !this.xray, depthWrite: !this.xray});
    const group = new THREE.Group();
    const fallback = new THREE.Mesh(this.capsule, material);
    const skeleton = new THREE.Group();
    const joints = Array.from({length: 15}, () => new THREE.Mesh(this.sphere, material));
    const bones = LIMBS.map(() => new THREE.Mesh(this.bone, material));
    for (const mesh of [fallback, ...joints, ...bones]) mesh.renderOrder = 10;
    skeleton.add(...joints, ...bones); group.add(fallback, skeleton); this.group.add(group);
    return {group, material, fallback, skeleton, joints, bones};
  }
  apply(scene) {
    if (scene?.schema_version !== 1 || !Array.isArray(scene.people)) return null;
    const active = new Set(); const parsed = [];
    for (const raw of scene.people) {
      const person = parsePerson(raw);
      if (!person || active.has(person.id)) continue;
      active.add(person.id); parsed.push(person);
      let state = this.people.get(person.id);
      if (!state) { state = this.create(person.id); this.people.set(person.id, state); }
      state.group.position.fromArray(person.root);
      state.fallback.visible = !person.joints;
      state.skeleton.visible = !!person.joints;
      if (person.joints) {
        state.joints.forEach((joint, i) => joint.position.fromArray(person.joints[i]));
        state.bones.forEach((bone, i) => {
          const [a, b] = LIMBS[i].map(j => state.joints[j].position);
          const delta = new THREE.Vector3().subVectors(b, a);
          const length = delta.length(); bone.visible = length > 1e-6;
          if (!bone.visible) return;
          bone.position.copy(a).add(b).multiplyScalar(.5);
          bone.scale.y = length; bone.quaternion.setFromUnitVectors(this.axis, delta.normalize());
        });
      }
    }
    for (const [id, state] of this.people) {
      if (!active.has(id)) { this.group.remove(state.group); state.material.dispose(); this.people.delete(id); }
    }
    this.group.visible = true;
    return parsed;
  }
  setXray(enabled) {
    this.xray = enabled;
    for (const state of this.people.values()) { state.material.depthTest = !enabled; state.material.depthWrite = !enabled; }
  }
  clear() { this.apply({schema_version: 1, people: []}); }
  dispose() { this.clear(); this.sphere.dispose(); this.bone.dispose(); this.capsule.dispose(); }
}
