// SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
// SPDX-License-Identifier: LGPL-2.1-or-later
import test from 'node:test';
import assert from 'node:assert/strict';
import {RecordingControls} from './recording_controls.js';

const scene = timestamp => ({schema_version: 1, coordinate_system: {frame: 'USD world', up_axis: 'Z', unit: 'millimetre'}, timestamp, people: []});
const fixture = () => new Blob([0, 1, 2].map(t => JSON.stringify(scene(t)) + '\n'));
const settle = () => new Promise(resolve => setTimeout(resolve, 10));
function setup(t) {
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, {value: id === 'play-speed' ? '1' : '0', checked: false, classList: {toggle() {}}});
    return elements.get(id);
  };
  const previous = globalThis.document;
  const listeners = new Map();
  globalThis.document = {getElementById: get, hidden: false,
    addEventListener: (name, callback) => listeners.set(name, callback),
    removeEventListener: name => listeners.delete(name),
    dispatchEvent: event => listeners.get(event.type)?.(),
  };
  const scenes = [], modes = [], statuses = [];
  const controls = new RecordingControls({onScene: s => scenes.push(s), onMode: m => modes.push(m), onStatus: s => statuses.push(s)});
  t.after(() => {
    controls.close();
    if (previous === undefined) delete globalThis.document; else globalThis.document = previous;
  });
  return {controls, scenes, modes, statuses, get};
}

test('file opens paused, seek changes scene, and live clears/cancels playback state', async t => {
  const {controls: c, scenes, modes, get} = setup(t);
  c.start(new URLSearchParams()); assert.deepEqual(modes, ['live']);
  await c.load('test.jsonl', async () => fixture()); await settle();
  assert.equal(c.mode, 'file'); assert.equal(c.player.playing, false); assert.equal(scenes.at(-1).timestamp, 0);
  c.seek(2); await settle(); assert.equal(scenes.at(-1).timestamp, 2);
  c.seek(0); await settle(); assert.equal(scenes.at(-1).timestamp, 0);
  get('play-pause').onclick(); c.tick(1); await settle(); assert.equal(scenes.at(-1).timestamp, 1);
  c.live(); assert.equal(c.player, null); assert.equal(get('playback-controls').disabled, true);
  assert.equal(get('mode-tag').textContent, 'LIVE'); assert.deepEqual(modes, ['live', 'file', 'live']);
});

test('cancelled loads and stale async frames cannot overwrite live/new file mode', async t => {
  const {controls: c, scenes} = setup(t);
  let finish;
  const pending = c.load('slow.jsonl', () => new Promise(resolve => { finish = resolve; }));
  c.live(); finish(fixture()); await pending; assert.equal(c.mode, 'live'); assert.equal(scenes.length, 0);
  await c.load('file.jsonl', async () => fixture()); await settle();
  let frame;
  c.player.recording.frame = () => new Promise(resolve => { frame = resolve; });
  c.seek(2); const count = scenes.length; c.live(); frame(scene(2)); await settle();
  assert.equal(scenes.length, count); assert.equal(c.pending, false);
});

test('invalid file has clear error, does not fall back to fake live scenes, and permits recovery', async t => {
  const {controls: c, get, scenes} = setup(t);
  await c.load('bad.jsonl', async () => new Blob(['{}\n']));
  assert.match(get('recording-warning').textContent, /schema/);
  assert.equal(c.mode, 'file'); assert.equal(c.player, null); assert.equal(scenes.length, 0);
  await c.load('valid.jsonl', async () => fixture()); await settle(); assert.equal(c.player.recording.count, 3);
  assert.equal(get('recording-warning').textContent, '');
});

test('paused scenes survive long waits and hidden tabs do not skip ahead', async t => {
  const {controls: c, scenes} = setup(t);
  await c.load('test.jsonl', async () => fixture()); await settle(); c.tick(60);
  assert.equal(c.player.index, 0); assert.equal(scenes.at(-1).timestamp, 0);
  // Browsers may suspend requestAnimationFrame entirely in a hidden tab.
  c.player.play(); document.hidden = true; document.dispatchEvent({type: 'visibilitychange'});
  document.hidden = false; document.dispatchEvent({type: 'visibilitychange'}); c.tick(60);
  assert.equal(c.player.index, 0); c.tick(1); assert.equal(c.player.index, 1);
});
