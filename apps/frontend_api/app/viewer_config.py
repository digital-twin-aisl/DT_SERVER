import os


def _env_int(name: str, default: int) -> int:
    """Read a TCP/UDP port from the environment without breaking app startup."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if 1 <= value <= 65535 else default


def viewer_config() -> dict:
    """Return browser-facing Isaac Sim settings.

    An empty public URL is intentional: the browser then uses the hostname from
    which the dashboard was opened. This makes LAN and mesh-VPN deployments work
    without baking a server address into the image.
    """
    public_url = os.getenv("ISAAC_WEBRTC_PUBLIC_URL", "").strip().rstrip("/")
    if public_url and not public_url.startswith(("http://", "https://")):
        public_url = ""

    path = os.getenv(
        "ISAAC_WEBRTC_PATH", "/streaming/webrtc-demo/"
    ).strip()
    if not path.startswith("/"):
        path = f"/{path}"

    return {
        "enabled": os.getenv("ISAAC_WEBRTC_ENABLED", "true").lower()
        not in {"0", "false", "no", "off"},
        "public_url": public_url or None,
        "server": os.getenv("ISAAC_WEBRTC_SERVER", "").strip() or None,
        "web_port": _env_int("ISAAC_WEBRTC_WEB_PORT", 8211),
        "signal_port": _env_int("ISAAC_WEBRTC_SIGNAL_PORT", 49100),
        "stream_port": _env_int("ISAAC_WEBRTC_STREAM_PORT", 47998),
        "path": path,
    }
