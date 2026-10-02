# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Long-running discovery, supervision and management API in one service."""

import fcntl
import os
import signal
import threading
import time

import uvicorn
import zenoh

from .api import create_app
from .control import ControlPlane
from .protocol import (
    EdgeTopics,
    camera_records_from_message,
    decode_json,
    encode_json,
    make_zenoh_config,
    validate_edge_id,
    heartbeat_command,
)
from .topology import RegionCatalog
from .transport import EdgeTransport


def serve(args, registry):
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.getenv(
        "DT_MANAGER_TOKEN"
    ):
        raise ValueError("set DT_MANAGER_TOKEN to expose the manager beyond localhost")
    if args.offline_after <= 0 or args.heartbeat_interval <= 0:
        raise ValueError("heartbeat and offline timeouts must be positive")
    registry.path.parent.mkdir(parents=True, exist_ok=True)
    # A single process owns the registry, HTTP API and GPU children.
    with (registry.path.parent / "manager.lock").open("w") as ownership:
        try:
            fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another manager already owns this registry") from exc
        _serve(args, registry)


def _serve(args, registry):
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    catalog = RegionCatalog(args.catalog)
    if any(
        region["topic_root"] != args.topic_root.strip("/") for region in catalog.all()
    ):
        raise ValueError("manager topic root must match all registered deployments")
    registry.mark_all_offline()
    with zenoh.open(make_zenoh_config(args.endpoint, args.zenoh_config)) as session:
        transport = EdgeTransport(session, args.topic_root)
        control = ControlPlane(
            catalog, registry, transport, args.data_dir, args.endpoint
        )

        def receive(sample):
            try:
                message = decode_json(sample.payload)
                edge_id = validate_edge_id(str(message["edge_id"]))
                topics = EdgeTopics(edge_id, args.topic_root)
                if (
                    message.get("kind") == "status"
                    and str(sample.key_expr) == topics.status
                ):
                    if not isinstance(message.get("data"), dict):
                        raise ValueError("status data must be an object")
                    if not isinstance(message["data"].get("inference", {}), dict):
                        raise ValueError("inference status must be an object")
                    registry.observe_status(edge_id, message)
                elif (
                    message.get("kind") == "camera_status"
                    and str(sample.key_expr) == topics.cameras
                ):
                    registry.observe_cameras(
                        edge_id,
                        camera_records_from_message(message),
                        edge_sent_at=message.get("sent_at"),
                    )
                elif (
                    message.get("kind") in {"command_ack", "config_ack"}
                    and str(sample.key_expr) == topics.ack
                ):
                    transport.on_ack(sample)
                    registry.record_ack(edge_id, message)
            except Exception as exc:
                print(f"Ignored invalid edge message: {exc}", flush=True)

        subscribers = [
            session.declare_subscriber(
                f"{args.topic_root.strip('/')}/*/{suffix}", receive
            )
            for suffix in ("status", "cameras", "ack")
        ]
        scene_subscribers = {}
        api = uvicorn.Server(
            uvicorn.Config(
                create_app(control), host=args.host, port=args.port, log_level="info"
            )
        )
        api_thread = threading.Thread(target=api.run, name="manager-http", daemon=True)
        api_thread.start()
        last_heartbeat = 0.0
        try:
            while not stop.wait(0.5):
                if not api_thread.is_alive():
                    raise RuntimeError("management HTTP server stopped")
                regions = catalog.all()
                active_topics = {region["scene_topic"] for region in regions}
                for topic in list(scene_subscribers):
                    if topic not in active_topics:
                        scene_subscribers.pop(topic).undeclare()
                for region in regions:
                    topic = region["scene_topic"]
                    if topic not in scene_subscribers:
                        scene_subscribers[topic] = session.declare_subscriber(
                            topic,
                            lambda sample: control.receive_scene(
                                str(sample.key_expr), sample.payload.to_bytes()
                            ),
                        )
                registry.mark_offline(args.offline_after)
                control.tick()
                if time.monotonic() - last_heartbeat >= args.heartbeat_interval:
                    for edge_id, record in registry.snapshot()["edges"].items():
                        if record.get("online"):
                            session.put(
                                EdgeTopics(edge_id, args.topic_root).command,
                                encode_json(heartbeat_command(edge_id)),
                            )
                    last_heartbeat = time.monotonic()
        finally:
            api.should_exit = True
            api_thread.join(timeout=10)
            control.close()
            for subscriber in [*subscribers, *scene_subscribers.values()]:
                subscriber.undeclare()
