"""Browser URLs are same-origin so one HTTP(S) tunnel also carries WebSockets."""
import os
from pathlib import Path


def assets_dir():
    return Path(os.getenv('VIEWER_ASSETS_DIR', str(Path(__file__).resolve().parents[1] / 'assets')))


def viewer_config():
    return {'renderer': 'threejs', 'enabled': True, 'path': '/viewer',
            'map_url': '/assets/map.glb', 'metadata_url': '/assets/map.json',
            'websocket_path': '/ws/scene', 'stale_seconds': 5.0}
