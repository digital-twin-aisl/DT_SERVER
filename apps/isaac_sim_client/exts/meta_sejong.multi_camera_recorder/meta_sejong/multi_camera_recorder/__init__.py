"""Isaac Sim multi-camera recorder extension package."""

try:
    from .extension import MultiCameraRecorderExtension
except ModuleNotFoundError as exc:
    # Keep the hard-coded calibration helpers testable with normal Python.
    if exc.name != "omni":
        raise
    MultiCameraRecorderExtension = None

__all__ = ["MultiCameraRecorderExtension"]
