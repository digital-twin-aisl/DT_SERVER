// SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
// SPDX-License-Identifier: LGPL-2.1-or-later
import test from 'node:test';
import assert from 'node:assert/strict';
import {SceneRecording, Playback, validateScene, recordingDownloadPath, downloadRecording, MAX_RECORDING_BYTES} from './playback.js';

const scene = (timestamp = 0, people = []) => ({schema_version: 1,
  coordinate_system: {frame: 'USD world', up_axis: 'Z', unit: 'millimetre'}, timestamp, people});
const person = {global_id: 7, root: {position: [1000, 2000, 3000], edge_id: 'edge_1'}, pose: null};
const blob = scenes => new Blob([scenes.map(s => JSON.stringify(s) + '\n').join('')]);

test('roundtrip schema, poses, empty snapshots and runtime data', async () => {
  const pose = {joint_format: 'voxelpose_15j_xyz', joints: Array.from({length: 15}, () => [1, 2, 3])};
  const scenes = [scene(123, [{...person, pose}]), {...scene(124), runtime: {playback: true}}, scene(125, [person])];
  const r = await SceneRecording.load(blob(scenes));
  assert.equal(r.count, 3); assert.equal(r.duration, 2);
  for (let i = 0; i < 3; i++) assert.deepEqual(await r.frame(i), scenes[i]);
  await assert.rejects(r.frame(3)); await assert.rejects(r.frame(-1));
});

test('byte offsets survive UTF-8, CRLF, blank lines and chunk boundaries', async () => {
  const s = {...scene(1), runtime: {note: '한글'.repeat(200_000)}};
  const r = await SceneRecording.load(new Blob(['\r\n', JSON.stringify(s), '\r\n  \n', JSON.stringify(scene(2))]));
  assert.equal(r.count, 2); assert.deepEqual(await r.frame(0), s); assert.deepEqual(await r.frame(1), scene(2));
});

test('ignore interrupted final JSON, but reject malformed complete/middle records', async () => {
  const r = await SceneRecording.load(new Blob([JSON.stringify(scene()), '\n{"timestamp":']));
  assert.equal(r.count, 1); assert.match(r.warning, /마지막 행/);
  await assert.rejects(SceneRecording.load(new Blob([JSON.stringify(scene()), '\nBROKEN\n'])), /JSON/);
  await assert.rejects(SceneRecording.load(new Blob(['BROKEN\n', JSON.stringify(scene())])), /JSON/);
  await assert.rejects(SceneRecording.load(new Blob(['{"timestamp":'])), /완전한 장면/);
});

test('reject unsupported schema/coordinates and invalid people rather than silently drawing wrong poses', () => {
  for (const bad of [{...scene(), schema_version: 2}, {...scene(), coordinate_system: {unit: 'metre'}},
    {...scene(), timestamp: Infinity}, {...scene(), people: null}, scene(0, [person, person]),
    scene(0, [{...person, global_id: -1}]), scene(0, [{...person, root: {position: [1, null, 3]}}]),
    scene(0, [{...person, pose: {joint_format: 'other', joints: []}}]), scene(0, [null])])
    assert.throws(() => validateScene(bad));
});

test('reject schema-invalid final JSON even without newline', async () => {
  await assert.rejects(SceneRecording.load(new Blob([JSON.stringify(scene()), '\n{}'])), /schema/);
});

test('reject backwards timestamps; duplicate times are allowed', async () => {
  await assert.rejects(SceneRecording.load(blob([scene(2), scene(1)])), /역순/);
  const r = await SceneRecording.load(blob([scene(1), scene(1), scene(2)]));
  assert.equal(r.count, 3);
});

test('empty file and oversized input/line have explicit errors', async () => {
  await assert.rejects(SceneRecording.load(new Blob([' \n\n'])), /완전한 장면/);
  await assert.rejects(SceneRecording.load({size: MAX_RECORDING_BYTES + 1}), /1 GiB/);
  await assert.rejects(SceneRecording.load(new Blob(['x'.repeat(16 * 1024 * 1024 + 1)])), /16 MiB/);
});

test('indexing can be aborted before/during load', async () => {
  const controller = new AbortController(); controller.abort();
  await assert.rejects(SceneRecording.load(blob([scene()]), {signal: controller.signal}), {name: 'AbortError'});
  const during = new AbortController();
  await assert.rejects(SceneRecording.load(blob([scene()]), {
    signal: during.signal, onProgress: () => during.abort(),
  }), {name: 'AbortError'});
});

test('playback uses relative timestamps for Unix and dataset clocks, and skips to the latest frame', async () => {
  for (const origin of [0, 1723456789]) {
    const p = new Playback(await SceneRecording.load(blob([0, .5, 1, 2].map(t => scene(origin + t)))));
    p.play(); p.advance(.75); assert.equal(p.index, 1);
    p.advance(1); assert.equal(p.index, 2);
    p.advance(10); assert.equal(p.index, 3); assert.equal(p.playing, false);
    p.play(); assert.equal(p.index, 0); assert.equal(p.elapsed, 0);
  }
});

test('pause, speed, backwards seek, loop and clamped seeks', async () => {
  const p = new Playback(await SceneRecording.load(blob([0, 1, 2].map(t => scene(t)))));
  p.advance(9); assert.equal(p.index, 0);
  p.speed = 2; p.play(); p.advance(.5); assert.equal(p.index, 1);
  p.playing = false; p.advance(9); assert.equal(p.index, 1);
  p.seek(-5); assert.equal(p.index, 0); p.seek(50); assert.equal(p.index, 2);
  p.seek(0); p.loop = true; p.play(); p.advance(1.25);
  assert.equal(p.elapsed, .5); assert.equal(p.index, 0); assert.equal(p.playing, true);
  assert.throws(() => p.advance(NaN)); assert.throws(() => p.advance(-1));
  p.speed = 0; assert.throws(() => p.advance(1));
});

test('one-frame and all-equal-timestamp recordings stop on the final scene, even in loop mode', async () => {
  for (const count of [1, 3]) {
    const p = new Playback(await SceneRecording.load(blob(Array.from({length: count}, () => scene(42)))));
    p.loop = true; p.play(); p.advance(0);
    assert.equal(p.index, count - 1); assert.equal(p.playing, false);
  }
});

test('recording URL is a read-only same-origin route, not arbitrary URLs or traversal', () => {
  assert.equal(recordingDownloadPath('region_1', '2026_run-1'), '/api/v1/control/regions/region_1/recordings/2026_run-1/download');
  for (const id of [null, '', '../secret', '..', '/etc/passwd', 'https://bad', 'a%2fb', 'a?token=x']) {
    assert.throws(() => recordingDownloadPath(id, 'run'));
    assert.throws(() => recordingDownloadPath('region', id));
  }
});

test('download validates HTTP errors, limits and reads streaming responses', async t => {
  const scenes = blob([scene(1)]);
  t.mock.method(globalThis, 'fetch', async () => new Response(scenes));
  assert.deepEqual(await (await SceneRecording.load(await downloadRecording('/test'))).frame(0), scene(1));
  globalThis.fetch = async () => new Response('', {status: 404});
  await assert.rejects(downloadRecording('/test'), /HTTP 404/);
  globalThis.fetch = async () => new Response('', {headers: {'content-length': String(MAX_RECORDING_BYTES + 1)}});
  await assert.rejects(downloadRecording('/test'), /1 GiB/);
});
