import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { PeopleRenderer } from './scene.js';

const $ = id => document.getElementById(id);
const toView = p => new THREE.Vector3(p[0], p[2], -p[1]);
const showError = message => { $('error').textContent = message; $('error').hidden = false; };
async function json(url) { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json(); }

async function main() {
  const config = await json('/api/v1/viewer/config');
  const metadata = await json(config.metadata_url);
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#a7bdc8');
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 4000);
  const renderer = new THREE.WebGLRenderer({antialias: true, powerPreference: 'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.2;
  $('viewport').append(renderer.domElement);
  let renderDirty = true;
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.addEventListener('change', () => { renderDirty = true; });
  controls.enableDamping = true; controls.maxPolarAngle = Math.PI * .49; controls.minDistance = 1; controls.maxDistance = 1700;
  scene.add(new THREE.HemisphereLight(0xd9eeff, 0x66715b, 2.5));
  const sun = new THREE.DirectionalLight(0xfff1d7, 3); sun.position.set(150, 250, 100); scene.add(sun);
  const world = new THREE.Group(); world.rotation.x = -Math.PI / 2; scene.add(world);
  const people = new PeopleRenderer(metadata.people_matrix); world.add(people.group);
  $('xray').onchange = e => { people.setXray(e.target.checked); renderDirty = true; };
  const zones = new THREE.Group(); world.add(zones);
  const deployment = metadata.deployment;
  for (const [i, edge] of (deployment?.edges || []).entries()) {
    const points = edge.polygon_m.map(p => new THREE.Vector3(...p));
    if (points.length) {
      const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({color: i ? 0xffc975 : 0x37f5da, depthTest: false, depthWrite: false}));
      line.renderOrder = 9; zones.add(line);
    }
  }
  for (const cam of deployment?.cameras || []) {
    const marker = new THREE.Mesh(new THREE.SphereGeometry(.16, 8, 6), new THREE.MeshBasicMaterial({color: 0xffd278}));
    marker.position.fromArray(cam.position_m); zones.add(marker);
    const m = cam.camera_to_world;
    const end = marker.position.clone().add(new THREE.Vector3(m[0][2], m[1][2], m[2][2]).multiplyScalar(1.5));
    zones.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([marker.position, end]), new THREE.LineBasicMaterial({color: 0xffd278})));
  }
  $('show-zones').onchange = e => { zones.visible = e.target.checked; renderDirty = true; };
  let following = false;
  const center = metadata.bounds_m[0].map((v, i) => (v + metadata.bounds_m[1][i]) / 2);
  const focus = deployment?.focus_m || center;
  function lookAt(point, distance) { const target = toView(point); controls.target.copy(target); camera.position.copy(target).add(new THREE.Vector3(-distance * .45, distance * 1.4, -distance * .65)); controls.update(); }
  function stopFollow() { following = false; $('follow').classList.remove('active'); }
  $('focus').onclick = () => { stopFollow(); lookAt(focus, 27); };
  $('campus').onclick = () => { stopFollow(); lookAt(center, 450); };
  $('follow').onclick = () => { following = !following; $('follow').classList.toggle('active', following); };
  lookAt(focus, deployment ? 27 : 450);
  const resize = () => { const {width, height} = $('viewport').getBoundingClientRect(); if (!width || !height) return; renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix(); renderDirty = true; };
  new ResizeObserver(resize).observe($('viewport')); resize();

  let latest = null, lastScene = 0, lastFrame = performance.now(), frames = 0, fpsTime = lastFrame;
  let parsed = [], socket, retry = 1000, retryTimer, closed = false;
  function status(text, live = false) { $('connection').textContent = text; $('dot').classList.toggle('live', live); }
  function counts() {
    $('people-count').textContent = parsed.length;
    $('pose-count').textContent = parsed.filter(p => p.joints).length;
    $('people-list').replaceChildren(...parsed.slice(0, 30).map(p => {
      const row = document.createElement('div'); row.className = 'person-row';
      const id = document.createElement('span'); id.textContent = `#${p.id} · ${p.edge || '—'}`;
      const mode = document.createElement('span'); mode.textContent = p.joints ? '15개 관절' : '위치 추적'; row.append(id, mode); return row;
    }));
  }
  function connect() {
    const url = new URL(config.websocket_path, location.href); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(url);
    socket.onopen = () => { retry = 1000; status('서버 장면 대기 중'); };
    socket.onmessage = e => {
      try {
        const data = JSON.parse(e.data);
        if (data.schema_version === 1 && Array.isArray(data.people)) { latest = data; lastScene = performance.now(); status('실시간 수신 중', true); }
        else if (data.type === 'status' && data.status !== 'live') { status(data.subscriber_ready ? '서버 장면 대기 중' : '추론 연결 대기 중'); }
      } catch { showError('서버 장면 형식을 읽을 수 없습니다.'); }
    };
    socket.onclose = () => { if (closed) return; status('연결 재시도 중'); retryTimer = setTimeout(connect, retry); retry = Math.min(retry * 2, 15000); };
    socket.onerror = () => socket.close();
  }
  connect();
  const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
  loader.load(config.map_url, gltf => {
    scene.add(gltf.scene); renderDirty = true;
    $('map-status').textContent = `세종대학교 · 캠퍼스 맵 준비됨`;
    // Expose read-only diagnostics for deployment smoke tests, no fake scene injection.
    document.body.dataset.mapReady = 'true';
  }, event => {
    $('map-status').textContent = event.total ? `캠퍼스 맵 ${Math.round(event.loaded / event.total * 100)}%` : `캠퍼스 맵 ${(event.loaded / 1048576).toFixed(1)} MB`;
  }, error => { $('map-status').textContent = '맵 로딩 실패'; showError(`맵을 불러오지 못했습니다. 새로고침해 다시 시도하세요. ${error.message}`); });

  renderer.setAnimationLoop(now => {
    if (latest) { renderDirty = true; const result = people.apply(latest); if (result) { parsed = result; counts(); } latest = null; }
    const age = lastScene ? (now - lastScene) / 1000 : null;
    if (age !== null && age > config.stale_seconds) {
      if (parsed.length) { people.clear(); parsed = []; counts(); renderDirty = true; }
      status('장면 수신 지연');
    }
    $('source-status').textContent = age === null ? '서버의 첫 장면을 기다립니다.' : `마지막 수신 ${age.toFixed(1)}초 전`;
    if (following && parsed.length) {
      const point = people.group.localToWorld(new THREE.Vector3(...parsed[0].root));
      const delta = point.clone().sub(controls.target).multiplyScalar(1 - Math.exp(-Math.min(now-lastFrame, 100) / 200));
      controls.target.add(delta); camera.position.add(delta);
    }
    controls.update();
    if (!document.hidden && renderDirty) { renderer.render(scene, camera); renderDirty = false; frames++; }
    if (now - fpsTime > 1000) { $('fps').textContent = frames ? Math.round(frames * 1000 / (now - fpsTime)) : '대기'; frames = 0; fpsTime = now; }
    lastFrame = now;
  });
  addEventListener('pagehide', () => { closed = true; clearTimeout(retryTimer); socket?.close(); renderer.setAnimationLoop(null); people.dispose(); controls.dispose(); renderer.dispose(); });
}
main().catch(error => { $('connection').textContent = '뷰어 시작 실패'; showError(`뷰어를 시작할 수 없습니다. ${error.message}`); });
