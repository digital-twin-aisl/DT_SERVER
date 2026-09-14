"""Bounded, read-only Zenoh SceneOutput -> browser WebSocket fan-out."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time

logger = logging.getLogger(__name__)
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON constant: {value}')


def validate_scene(raw: bytes) -> dict:
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise ValueError('SceneOutput exceeds size limit')
    scene = json.loads(raw, parse_constant=_reject_constant)
    if not isinstance(scene, dict) or type(scene.get('schema_version')) is not int or scene.get('schema_version') != 1:
        raise ValueError('Expected SceneOutput schema_version=1')
    if not isinstance(scene.get('people'), list) or len(scene['people']) > 4096:
        raise ValueError('Invalid people array')
    timestamp = scene.get('timestamp')
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise ValueError('Invalid timestamp')
    return scene


class SceneBridge:
    def __init__(self, endpoint, topic, stale_seconds=5.0):
        self.endpoint = endpoint if '/' in endpoint else f'tcp/{endpoint}'
        self.topic = topic
        self.stale_seconds = stale_seconds
        self.clients = set()
        self._lock = threading.Lock()
        self._pending = None
        self._latest = None
        self._last_received = None
        self._stop = threading.Event()
        self._thread = None
        self._task = None
        self.connected = False
        self.error = None
        self.received = self.invalid = self.dropped = 0

    def receive(self, raw):
        try:
            validate_scene(raw)
            text = raw.decode('utf-8')
        except (ValueError, UnicodeError, RecursionError):
            with self._lock: self.invalid += 1
            return
        with self._lock:
            if self._pending is not None: self.dropped += 1
            self._pending = text
            self._latest = text
            self._last_received = time.monotonic()
            self.received += 1

    def _run(self):
        while not self._stop.is_set():
            try:
                import zenoh
                config = zenoh.Config.from_json5(json.dumps({
                    'mode': 'client', 'connect': {'endpoints': [self.endpoint], 'timeout_ms': 3000},
                    'scouting': {'multicast': {'enabled': False}}}))
                with zenoh.open(config) as session:
                    subscriber = session.declare_subscriber(self.topic, lambda sample: self.receive(sample.payload.to_bytes()))
                    self.connected, self.error = True, None
                    logger.info('Scene subscriber ready: %s %s', self.endpoint, self.topic)
                    self._stop.wait()
                    subscriber.undeclare()
            except Exception as exc:
                self.error = str(exc)
                logger.warning('Scene subscriber reconnecting: %s', exc)
                self._stop.wait(3)
            finally:
                self.connected = False

    async def start(self):
        self._thread = threading.Thread(target=self._run, name='scene-zenoh', daemon=True)
        self._thread.start()
        self._task = asyncio.create_task(self._fanout())

    async def close(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self._thread:
            await asyncio.to_thread(self._thread.join, 5)

    async def _fanout(self):
        # Poll one slot at <=30Hz: producer threads never enqueue unbounded loop callbacks.
        while True:
            await asyncio.sleep(1 / 30)
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is None: continue
            for queue in tuple(self.clients):
                if queue.full():
                    queue.get_nowait()
                    self.dropped += 1
                queue.put_nowait(pending)

    def subscribe(self):
        queue = asyncio.Queue(maxsize=1)
        self.clients.add(queue)
        with self._lock:
            if self._latest is not None and time.monotonic() - self._last_received < self.stale_seconds:
                queue.put_nowait(self._latest)
        return queue

    def status(self):
        with self._lock:
            age = None if self._last_received is None else time.monotonic() - self._last_received
            return {'status': 'live' if age is not None and age < self.stale_seconds else 'waiting',
                    'transport': 'zenoh', 'subscriber_ready': self.connected, 'topic': self.topic,
                    'last_scene_age_seconds': age, 'received': self.received, 'invalid': self.invalid,
                    'dropped': self.dropped, 'connections': len(self.clients), 'error': self.error}
