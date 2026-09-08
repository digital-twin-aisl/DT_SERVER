"""Shared bounded publishers; successful put counts enqueue, not delivery ACKs."""

import json
import queue
import threading

import zenoh


def make_zenoh_config(endpoint=None, config_path=None):
    if config_path:
        return zenoh.Config.from_file(config_path)
    if endpoint:
        endpoint = endpoint if "/" in endpoint else f"tcp/{endpoint}"
        return zenoh.Config.from_json5(
            json.dumps({"mode": "client", "connect": {"endpoints": [endpoint]}})
        )
    return zenoh.Config()


class LatestPublisher:
    def __init__(self, topic, endpoint=None, config_path=None, queue_size=2):
        if not topic or queue_size < 1:
            raise ValueError("publisher requires a topic and a positive queue size")
        self.topic, self.endpoint, self.config_path = topic, endpoint, config_path
        self.queue = queue.Queue(queue_size)
        self.dropped = self.sent_messages = self.sent_bytes = 0
        self.closed = False
        self._error = None
        self._session = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.worker = threading.Thread(
            target=self._run, name="zenoh-publisher", daemon=True
        )
        self.worker.start()
        if not self._ready.wait(10):
            self.close()
            raise TimeoutError("Zenoh publisher initialization timed out")
        if self._error is not None:
            self.close()
            raise RuntimeError("Zenoh publisher initialization failed") from self._error

    @property
    def router_ids(self):
        try:
            return (
                tuple(str(value) for value in self._session.info.routers_zid())
                if self._session
                else ()
            )
        except Exception:
            return ()

    @property
    def link_count(self):
        try:
            return len(self._session.info.links()) if self._session else 0
        except Exception:
            return 0

    @property
    def router_connected(self):
        return self.link_count > 0

    def _run(self):
        try:
            with zenoh.open(
                make_zenoh_config(self.endpoint, self.config_path)
            ) as session:
                self._session = session
                publisher = session.declare_publisher(
                    self.topic, congestion_control=zenoh.CongestionControl.DROP
                )
                self._ready.set()
                while not self._stop.is_set():
                    try:
                        payload = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    publisher.put(payload)
                    self.sent_messages += 1
                    self.sent_bytes += len(payload)
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._session = None

    def send(self, payload):
        if self.closed or self._error is not None:
            raise RuntimeError("Zenoh publisher is closed or failed") from self._error
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(payload)
            except queue.Full:
                self.dropped += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._stop.set()
        if threading.current_thread() is not self.worker:
            self.worker.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
