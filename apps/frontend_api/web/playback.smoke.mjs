// SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
// SPDX-License-Identifier: LGPL-2.1-or-later
// Optional real-browser test. No manager, workers, router, or campus assets needed.
// PLAYWRIGHT_MODULE may point to an installed playwright(-core) index.mjs.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const staticRoot = new URL('../app/static/', import.meta.url);
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const person = {global_id: 9, lod: 0, root: {position: [0, 0, 1000], edge_id: 'edge_1'}, pose: null};
const posePerson = {...person, pose: {joint_format: 'voxelpose_15j_xyz', joints: Array.from({length: 15}, (_, i) => [i * 30, 0, 1000 + i * 60])}};
const scene = (timestamp, people) => ({schema_version: 1, coordinate_system: {frame: 'USD world', up_axis: 'Z', unit: 'millimetre'}, timestamp, people});
const recording = [scene(100, [person]), scene(101, [posePerson]), scene(102, [])].map(s => JSON.stringify(s) + '\n').join('');
const region = {region_id: 'region_1', name: '테스트 구역', state: 'running', edges: [], last_scene_age_seconds: null};
const requests = [];
const server = createServer(async (req, res) => {
  requests.push({method: req.method, url: req.url});
  const path = new URL(req.url, 'http://localhost').pathname;
  const json = data => { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(data)); };
  if (path === '/api/v1/viewer/config') return json({metadata_url: '/assets/map.json', map_url: '/assets/map.gltf', websocket_path: '/ws/scene', stale_seconds: .2});
  if (path === '/assets/map.json') return json({people_matrix: identity, bounds_m: [[-5, -5, 0], [5, 5, 5]], deployment: {focus_m: [0, 0, 0], edges: [], cameras: []}});
  if (path === '/assets/map.gltf') return json({asset: {version: '2.0'}, scene: 0, scenes: [{nodes: []}], nodes: []});
  if (path === '/api/v1/control/regions') return json([region]);
  if (path === '/api/v1/control/edges') return json([]);
  if (path === '/api/v1/control/regions/region_1/recordings') return json([{operation: 'inference', status: 'completed', started_at: 100, run_id: 'run_1', scene_bytes: recording.length}]);
  if (path === '/api/v1/control/regions/region_1/recordings/run_1/download') {
    res.setHeader('Content-Type', 'application/x-ndjson'); return res.end(recording);
  }
  if (path.startsWith('/api/')) { res.writeHead(503); return res.end('Unexpected API call'); }
  const files = new Map([['/', 'index.html'], ['/viewer', 'viewer.html'], ['/static/viewer.css', 'viewer.css'],
    ['/static/dist/viewer.js', 'dist/viewer.js'], ['/static/control.js', 'control.js'], ['/static/control.css', 'control.css']]);
  const name = files.get(path);
  if (!name) { res.writeHead(404); return res.end(); }
  try {
    res.setHeader('Content-Type', name.endsWith('.js') ? 'text/javascript' : name.endsWith('.css') ? 'text/css' : 'text/html');
    res.end(await readFile(new URL(name, staticRoot)));
  } catch (error) { res.writeHead(500); res.end(error.message); }
});
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
const base = `http://127.0.0.1:${server.address().port}`;
let browser;
try {
  browser = await chromium.launch({headless: true,
    ...(process.env.CHROME_BIN ? {executablePath: process.env.CHROME_BIN} : {}),
    args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  const page = await browser.newPage({viewport: {width: 1280, height: 900}});
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  let connections = 0, disconnected = 0;
  await page.routeWebSocket('**/ws/scene*', ws => {
    connections++; ws.send(JSON.stringify(scene(900, [person]))); ws.onClose(() => { disconnected++; });
  });
  const textIs = async (id, value) => page.waitForFunction(({id, value}) => document.getElementById(id).textContent === value, {id, value});
  const openFile = async (name = 'sample.jsonl', buffer = recording) => page.locator('#recording-file').setInputFiles({name, mimeType: 'application/x-ndjson', buffer: Buffer.from(buffer)});
  await page.goto(base + '/viewer');
  await textIs('people-count', '1'); assert.equal(connections, 1);
  await textIs('connection', '장면 수신 지연'); await textIs('people-count', '0');
  await openFile(); await textIs('connection', '파일 일시정지'); await textIs('people-count', '1');
  assert.equal(await page.locator('#mode-tag').textContent(), 'FILE');
  await page.waitForTimeout(400); assert.equal(await page.locator('#people-count').textContent(), '1');
  assert.equal(disconnected, 1);
  await page.locator('#frame-next').click(); await textIs('pose-count', '1');
  await page.locator('#frame-next').click(); await textIs('people-count', '0');
  await page.locator('#frame-prev').click(); await textIs('people-count', '1');
  await page.locator('#frame-first').click(); await textIs('pose-count', '0');
  await page.locator('#play-speed').selectOption('8'); await page.locator('#play-pause').click();
  await textIs('connection', '파일 재생 끝'); await textIs('people-count', '0');
  await page.locator('#frame-first').click(); await page.locator('#play-loop').check();
  await page.locator('#play-pause').click(); await page.waitForTimeout(500);
  assert.equal(await page.locator('#play-pause').textContent(), '일시정지');
  await page.locator('#play-pause').click();
  await openFile('invalid.jsonl', '{}\n'); await textIs('connection', '기록 불러오기 실패');
  assert.match(await page.locator('#recording-warning').textContent(), /schema/);
  await page.locator('#mode-live').click(); await textIs('people-count', '1'); assert.equal(connections, 2);
  await openFile(); await textIs('connection', '파일 일시정지');
  if (process.env.SMOKE_SCREENSHOT) await page.screenshot({path: process.env.SMOKE_SCREENSHOT});

  // Direct replay must work even when the manager's regional configuration endpoint is unavailable.
  const before = connections;
  await page.goto(base + '/viewer?region=region_1&recording=run_1');
  await textIs('connection', '파일 일시정지'); await textIs('people-count', '1');
  assert.equal(connections, before);
  assert.ok(requests.some(r => r.url.endsWith('/run_1/download')));
  assert.ok(!requests.some(r => r.url === '/api/v1/control/regions/region_1'));
  assert.ok(!requests.some(r => r.method !== 'GET'));
  await page.setViewportSize({width: 390, height: 844});
  assert.ok(await page.locator('#mode-live').isVisible());
  const box = await page.locator('.recording').boundingBox();
  assert.ok(box.x >= 0 && box.x + box.width <= 390 && box.y + box.height <= 844);
  await page.goto(base + '/');
  const preview = page.getByRole('link', {name: '뷰어 재생', exact: true}); await preview.waitFor();
  assert.equal(await preview.getAttribute('href'), '/viewer?region=region_1&recording=run_1');
  assert.ok(await page.getByRole('button', {name: '구역 재생', exact: true}).isVisible());
  assert.ok(await page.locator('#start').isEnabled());
  assert.deepEqual(errors, []);
  console.log('FRONTEND_PLAYBACK_BROWSER_SMOKE_OK (live / local file / seek / pause / loop / errors / reconnect / dashboard preview)');
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
