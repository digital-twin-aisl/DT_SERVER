# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Correlated requests over the manager's existing Zenoh session."""

import threading
import time
from uuid import uuid4

from dt_common.contracts.edge import EdgeTopics, decode_json, encode_json


class EdgeTransport:
    def __init__(self, session, topic_root="dt/edges", timeout=3):
        self.session, self.topic_root, self.timeout = session, topic_root, timeout
        self.pending = {}
        self.lock = threading.Lock()

    def on_ack(self, sample):
        try:
            message = decode_json(sample.payload)
            key = message.get("command_id") or message.get("config_id")
            with self.lock:
                item = self.pending.get(key)
                if not item:
                    return
                edge, expected_kind, event, response = item
                if (
                    message.get("edge_id") != edge
                    or message.get("kind") != expected_kind
                    or str(sample.key_expr) != EdgeTopics(edge, self.topic_root).ack
                ):
                    return
                response.update(message)
                event.set()
        except (ValueError, KeyError, TypeError):
            return

    def request(self, edge_id, command, parameters=None, *, configuration=False):
        key = uuid4().hex
        topics = EdgeTopics(edge_id, self.topic_root)
        event, response = threading.Event(), {}
        message = {"schema_version": 1, "edge_id": edge_id, "sent_at": time.time()}
        if configuration:
            message.update(kind="edge_config", config_id=key, data=parameters)
        else:
            message.update(
                kind="command",
                command_id=key,
                command=command,
                parameters=parameters or {},
                expires_at=time.time() + self.timeout + 5,
            )
        with self.lock:
            self.pending[key] = (
                edge_id,
                "config_ack" if configuration else "command_ack",
                event,
                response,
            )
        try:
            self.session.put(
                topics.config if configuration else topics.command, encode_json(message)
            )
            if not event.wait(self.timeout):
                raise TimeoutError(f"{edge_id}: {command} acknowledgement timed out")
            if not response.get("success"):
                raise ValueError(
                    f"{edge_id}: {response.get('error') or 'command rejected'}"
                )
            return response.get("result") or {}
        finally:
            with self.lock:
                self.pending.pop(key, None)
