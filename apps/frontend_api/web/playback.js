// SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
// SPDX-License-Identifier: LGPL-2.1-or-later
// SceneOutput JSONL: index byte ranges, keep poses on disk until displayed.
export const MAX_RECORDING_BYTES = 1024 * 1024 * 1024;
const MAX_LINE_BYTES = 16 * 1024 * 1024;
const MAX_FRAMES = 2_000_000;
const CHUNK_BYTES = 1024 * 1024;
const point = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);

export function validateScene(scene) {
  if (scene?.schema_version !== 1) throw new Error('SceneOutput schema_version=1 파일이 필요합니다.');
  const c = scene.coordinate_system;
  if (c?.frame !== 'USD world' || c.up_axis !== 'Z' || c.unit !== 'millimetre')
    throw new Error('USD world / Z-up / millimetre 좌표가 필요합니다.');
  if (!Number.isFinite(scene.timestamp)) throw new Error('유효하지 않은 장면 시간입니다.');
  if (!Array.isArray(scene.people) || scene.people.length > 1000) throw new Error('유효하지 않은 사람 목록입니다.');
  const ids = new Set();
  for (const p of scene.people) {
    if (!p || !Number.isSafeInteger(p.global_id) || p.global_id < 0 || ids.has(p.global_id))
      throw new Error('사람 ID는 중복 없는 0 이상의 정수여야 합니다.');
    ids.add(p.global_id);
    if (!point(p.root?.position)) throw new Error('유효하지 않은 root 좌표입니다.');
    if (p.pose != null && (p.pose.joint_format !== 'voxelpose_15j_xyz' ||
        !Array.isArray(p.pose.joints) || p.pose.joints.length !== 15 || !p.pose.joints.every(point)))
      throw new Error('자세에는 유효한 XYZ 관절 15개가 필요합니다.');
  }
  return scene;
}

export class SceneRecording {
  static async load(blob, {signal, onProgress = () => {}} = {}) {
    if (blob.size > MAX_RECORDING_BYTES) throw new Error('기록은 1 GiB 이하로 나누어 주세요.');
    const offsets = [], lengths = [], timestamps = [];
    const decoder = new TextDecoder('utf-8', {fatal: true});
    let tail = new Uint8Array(), line = 0, warning = '';
    function consume(bytes, offset, final = false) {
      line++;
      if (bytes.length > MAX_LINE_BYTES) throw new Error(`${line}행: 장면이 16 MiB를 초과합니다.`);
      let scene;
      try {
        const text = decoder.decode(bytes).trim();
        if (!text) return;
        scene = JSON.parse(text);
      } catch {
        if (final) { warning = `중단된 기록의 불완전한 마지막 행(${line})을 제외했습니다.`; return; }
        throw new Error(`${line}행: JSON을 읽을 수 없습니다.`);
      }
      try {
        validateScene(scene);
        if (timestamps.length && scene.timestamp < timestamps.at(-1))
          throw new Error('시간이 역순입니다. 실행별로 기록 파일을 분리해 주세요.');
      } catch (error) { throw new Error(`${line}행: ${error.message}`); }
      offsets.push(offset); lengths.push(bytes.length); timestamps.push(scene.timestamp);
      if (offsets.length > MAX_FRAMES) throw new Error('장면 200만 개 이하로 기록을 나누어 주세요.');
    }
    for (let start = 0; start < blob.size; start += CHUNK_BYTES) {
      signal?.throwIfAborted();
      const chunk = new Uint8Array(await blob.slice(start, start + CHUNK_BYTES).arrayBuffer());
      signal?.throwIfAborted();
      const base = start - tail.length;
      const data = new Uint8Array(tail.length + chunk.length); data.set(tail); data.set(chunk, tail.length);
      let from = 0, end;
      while ((end = data.indexOf(10, from)) !== -1) {
        consume(data.subarray(from, end), base + from); from = end + 1;
      }
      tail = data.slice(from);
      if (tail.length > MAX_LINE_BYTES) throw new Error(`${line + 1}행: 장면이 16 MiB를 초과합니다.`);
      onProgress(Math.min(start + chunk.length, blob.size) / blob.size);
      // Yield between chunks even when Blob reads are memory-backed.
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    signal?.throwIfAborted();
    if (tail.length) consume(tail, blob.size - tail.length, true);
    if (!offsets.length) throw new Error('완전한 장면이 없는 기록입니다.');
    return new SceneRecording(blob, offsets, lengths, timestamps, warning);
  }

  constructor(blob, offsets, lengths, timestamps, warning) {
    Object.assign(this, {blob, offsets, lengths, timestamps, warning});
  }
  get count() { return this.offsets.length; }
  get duration() { return this.timestamps.at(-1) - this.timestamps[0]; }
  async frame(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.count) throw new Error('장면 번호 범위 오류');
    const offset = this.offsets[index];
    const scene = validateScene(JSON.parse(await this.blob.slice(offset, offset + this.lengths[index]).text()));
    if (scene.timestamp !== this.timestamps[index]) throw new Error('기록 파일이 변경되었습니다. 다시 여세요.');
    return scene;
  }
}

export class Playback {
  constructor(recording) {
    this.recording = recording; this.index = 0; this.elapsed = 0;
    this.playing = false; this.speed = 1; this.loop = false;
  }
  seek(index) {
    this.index = Math.max(0, Math.min(Math.trunc(index), this.recording.count - 1));
    this.elapsed = this.recording.timestamps[this.index] - this.recording.timestamps[0];
  }
  play() { if (this.index === this.recording.count - 1) this.seek(0); this.playing = true; }
  advance(dt) {
    if (!Number.isFinite(dt) || dt < 0 || !Number.isFinite(this.speed) || this.speed < .1 || this.speed > 8)
      throw new Error('유효하지 않은 재생 시간 또는 배속입니다.');
    if (!this.playing) return;
    this.elapsed += dt * this.speed;
    const {duration, timestamps, count} = this.recording;
    if (this.elapsed >= duration) {
      if (this.loop && duration > 0) this.elapsed %= duration;
      else { this.seek(count - 1); this.playing = false; return; }
    }
    const stamp = timestamps[0] + this.elapsed;
    let low = 0, high = count;
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if (timestamps[mid] <= stamp) low = mid + 1; else high = mid;
    }
    this.index = Math.max(0, low - 1);
  }
}

// Build only the existing same-origin, read-only management download endpoint.
export function recordingDownloadPath(region, recording) {
  const valid = value => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(value);
  if (!valid(region) || !valid(recording)) throw new Error('유효하지 않은 구역 또는 기록 ID입니다.');
  return `/api/v1/control/regions/${encodeURIComponent(region)}/recordings/${encodeURIComponent(recording)}/download`;
}

export async function downloadRecording(url, {signal, onProgress = () => {}} = {}) {
  const response = await fetch(url, {signal});
  if (!response.ok) throw new Error(`기록 다운로드 실패: HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length'));
  if (total > MAX_RECORDING_BYTES) {
    await response.body?.cancel();
    throw new Error('기록은 1 GiB 이하로 나누어 주세요.');
  }
  const reader = response.body.getReader(), chunks = [];
  let size = 0;
  try {
    while (true) {
      signal?.throwIfAborted();
      const {value, done} = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_RECORDING_BYTES) throw new Error('기록은 1 GiB 이하로 나누어 주세요.');
      chunks.push(value); onProgress(size, total);
    }
    return new Blob(chunks, {type: 'application/x-ndjson'});
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}
