# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Read-only HTTP / WebSocket regression checks; no router or manager needed."""
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from apps.frontend_api.app.main import app
from apps.frontend_api.app.scene_bridge import SceneBridge


class RecordingViewerTests(unittest.TestCase):
    def setUp(self):
        # Disable the network subscriber, retain the real scene queue and routes.
        self.start = patch.object(SceneBridge, "start", new=AsyncMock())
        self.start.start()
        self.addCleanup(self.start.stop)
        # TestCase.enterContext is Python 3.11+; keep 3.10 compatible.
        self.client = TestClient(app).__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_default_live_configuration_and_routes_remain_available(self):
        result = self.client.get("/api/v1/viewer/config")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["websocket_path"], "/ws/scene")
        self.assertEqual(result.json()["stale_seconds"], 5.0)
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/viewer/status").json()["transport"], "zenoh")

    def test_recording_viewer_does_not_require_manager_to_serve_page_or_base_config(self):
        with patch("apps.frontend_api.app.main.region_bridge", new=AsyncMock(side_effect=AssertionError("Must not resolve region"))):
            page = self.client.get("/viewer?region=offline&recording=run_1")
            self.assertEqual(page.status_code, 200)
            self.assertIn('id="recording-file"', page.text)
            self.assertIn('id="mode-live"', page.text)
            self.assertEqual(self.client.get("/api/v1/viewer/config").status_code, 200)

    def test_live_websocket_still_sends_same_scene_and_releases_subscription(self):
        scene = {"schema_version": 1, "timestamp": 1.0, "people": [],
                 "coordinate_system": {"frame": "USD world", "up_axis": "Z", "unit": "millimetre"}}
        bridge = app.state.scene_bridge
        bridge.receive(json.dumps(scene).encode())
        with self.client.websocket_connect("/ws/scene") as socket:
            self.assertEqual(socket.receive_json(), scene)
            self.assertEqual(len(bridge.clients), 1)
        self.assertEqual(len(bridge.clients), 0)

    def test_dashboard_retains_start_and_record_controls(self):
        page = self.client.get("/")
        for name in ("start", "stop", "record", "view"):
            self.assertIn(f'id="{name}"', page.text)
        self.assertIn("JSONL 파일 뷰어", page.text)
        script = self.client.get("/static/control.js").text
        self.assertIn("뷰어 재생", script)
        self.assertIn("구역 재생", script)


if __name__ == "__main__":
    unittest.main()
