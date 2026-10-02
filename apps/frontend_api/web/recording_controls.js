// SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
// SPDX-License-Identifier: LGPL-2.1-or-later
import {SceneRecording, Playback, recordingDownloadPath, downloadRecording} from './playback.js';

const $ = id => document.getElementById(id);
const seconds = value => `${value.toFixed(2)}초`;

// File playback belongs to this viewer only; it never publishes or controls a worker.
export class RecordingControls {
  constructor({onMode, onScene, onStatus}) {
    Object.assign(this, {onMode, onScene, onStatus});
    this.mode = 'live'; this.player = null; this.generation = 0; this.pending = false; this.displayed = -1;
    this.visibilityChanged = () => { if (document.hidden) this.wasHidden = true; };
    document.addEventListener('visibilitychange', this.visibilityChanged);
    $('mode-live').onclick = () => this.live();
    $('recording-file').onchange = e => {
      const file = e.target.files[0];
      if (file) this.load(file.name, async () => file);
      e.target.value = ''; // Permit choosing the same file again.
    };
    $('play-pause').onclick = () => {
      if (!this.player) return;
      if (this.player.playing) this.player.playing = false; else this.player.play();
      this.update();
    };
    $('frame-first').onclick = () => this.seek(0);
    $('frame-prev').onclick = () => this.seek(this.player.index - 1);
    $('frame-next').onclick = () => this.seek(this.player.index + 1);
    $('frame-seek').oninput = e => this.seek(Number(e.target.value));
    $('play-speed').onchange = e => { if (this.player) this.player.speed = Number(e.target.value); };
    $('play-loop').onchange = e => { if (this.player) this.player.loop = e.target.checked; };
  }
  start(params) {
    if (params.has('recording')) {
      this.load(params.get('recording'), signal => downloadRecording(
        recordingDownloadPath(params.get('region'), params.get('recording')), {
          signal, onProgress: bytes => { $('recording-name').textContent = `기록 다운로드 ${(bytes / 1048576).toFixed(1)} MB…`; },
        }));
    } else this.live();
  }
  reset(mode) {
    this.abort?.abort(); this.generation++;
    this.mode = mode; this.player = null; this.pending = false; this.displayed = -1;
    this.wasHidden = document.hidden;
    $('playback-controls').disabled = true;
    $('mode-live').classList.toggle('active', mode === 'live');
    $('mode-tag').textContent = mode === 'live' ? 'LIVE' : 'FILE';
    $('playback-panel').hidden = mode === 'live';
    $('recording-warning').textContent = '';
    $('playback-time').textContent = '—';
    $('frame-seek').value = 0;
    this.onMode(mode);
  }
  live() {
    this.reset('live');
    $('recording-name').textContent = 'JSONL 파일은 브라우저에서만 열며 업로드하지 않습니다.';
  }
  async load(name, getBlob) {
    this.reset('file');
    this.abort = new AbortController();
    const {signal} = this.abort, generation = this.generation;
    this.onStatus('기록 불러오는 중');
    $('recording-name').textContent = name;
    try {
      const blob = await getBlob(signal);
      signal.throwIfAborted();
      const recording = await SceneRecording.load(blob, {
        signal, onProgress: fraction => { $('recording-name').textContent = `${name} · 검증 ${Math.round(fraction * 100)}%`; },
      });
      if (generation !== this.generation) return;
      this.player = new Playback(recording);
      this.player.speed = Number($('play-speed').value); this.player.loop = $('play-loop').checked;
      $('recording-name').textContent = name;
      $('recording-warning').textContent = recording.warning;
      $('frame-seek').max = recording.count - 1;
      $('playback-controls').disabled = false;
      this.update(); this.readFrame();
    } catch (error) {
      if (generation !== this.generation || signal.aborted) return;
      $('recording-warning').textContent = error.message;
      this.onStatus('기록 불러오기 실패');
    }
  }
  seek(index) {
    if (!this.player) return;
    this.player.playing = false; this.player.seek(index); this.update(); this.readFrame();
  }
  readFrame() {
    const player = this.player;
    if (!player || this.pending || this.displayed === player.index) return;
    const index = player.index, generation = this.generation;
    this.pending = true;
    player.recording.frame(index).then(scene => {
      if (generation !== this.generation || (!player.playing && player.index !== index)) return;
      this.displayed = index; this.onScene(scene);
    }).catch(error => {
      if (generation !== this.generation) return;
      player.playing = false; this.player = null;
      $('playback-controls').disabled = true;
      $('recording-warning').textContent = error.message; this.onStatus('기록 읽기 실패');
    }).finally(() => { if (generation === this.generation) this.pending = false; });
  }
  update() {
    if (!this.player) return;
    const p = this.player;
    $('play-pause').textContent = p.playing ? '일시정지' : '재생';
    $('frame-seek').value = p.index;
    $('playback-time').textContent = `${seconds(p.elapsed)} / ${seconds(p.recording.duration)} · 장면 ${p.index + 1} / ${p.recording.count}`;
    this.onStatus(p.playing ? '파일 재생 중' : p.index === p.recording.count - 1 ? '파일 재생 끝' : '파일 일시정지', p.playing);
    $('source-status').textContent = '파일 모드 · 서버 실행과 다른 뷰어에 영향을 주지 않습니다.';
  }
  tick(dt) {
    if (!this.player) return;
    // Background tabs do not silently skip ahead when brought to the foreground.
    if (document.hidden) { this.wasHidden = true; return; }
    this.player.advance(this.wasHidden ? 0 : dt); this.wasHidden = false;
    this.update(); this.readFrame();
  }
  close() {
    this.abort?.abort(); this.generation++; this.player = null;
    document.removeEventListener('visibilitychange', this.visibilityChanged);
  }
}
