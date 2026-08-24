import os
import unittest
from unittest.mock import patch

from apps.frontend_api.app.viewer_config import viewer_config


class ViewerConfigTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "ISAAC_WEBRTC_PUBLIC_URL": "https://viewer.example.com/base/",
            "ISAAC_WEBRTC_SERVER": "203.0.113.10",
            "ISAAC_WEBRTC_WEB_PORT": "18211",
            "ISAAC_WEBRTC_SIGNAL_PORT": "59100",
            "ISAAC_WEBRTC_STREAM_PORT": "57998",
            "ISAAC_WEBRTC_PATH": "streaming/webrtc-demo/",
            "ISAAC_WEBRTC_ENABLED": "true",
        },
        clear=True,
    )
    def test_explicit_remote_viewer_config(self):
        self.assertEqual(
            viewer_config(),
            {
                "enabled": True,
                "public_url": "https://viewer.example.com/base",
                "server": "203.0.113.10",
                "web_port": 18211,
                "signal_port": 59100,
                "stream_port": 57998,
                "path": "/streaming/webrtc-demo/",
            },
        )

    @patch.dict(
        os.environ,
        {
            "ISAAC_WEBRTC_PUBLIC_URL": "javascript:alert(1)",
            "ISAAC_WEBRTC_WEB_PORT": "not-a-port",
            "ISAAC_WEBRTC_SIGNAL_PORT": "70000",
            "ISAAC_WEBRTC_STREAM_PORT": "0",
            "ISAAC_WEBRTC_ENABLED": "off",
        },
        clear=True,
    )
    def test_invalid_values_fall_back_safely(self):
        config = viewer_config()
        self.assertFalse(config["enabled"])
        self.assertIsNone(config["public_url"])
        self.assertEqual(config["web_port"], 8211)
        self.assertEqual(config["signal_port"], 49100)
        self.assertEqual(config["stream_port"], 47998)


if __name__ == "__main__":
    unittest.main()
