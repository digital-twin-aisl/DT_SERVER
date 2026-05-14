"""
Anchor Utilities — GPS/UWB 절대 좌표 변환 헬퍼

지구상의 위경도(geodetic) 좌표를 로컬 ENU(East-North-Up) 좌표로 변환합니다.
GPS NavSatFix → world 좌표계 prior factor에 사용됩니다.

UWB는 일반적으로 이미 로컬 좌표계로 제공되므로 직접 prior로 주입합니다.
"""
import numpy as np
from typing import Optional, Tuple
import threading
import logging

logger = logging.getLogger(__name__)

# WGS84 ellipsoid constants
_WGS84_A = 6378137.0                  # semi-major axis (m)
_WGS84_F = 1.0 / 298.257223563        # flattening
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)  # eccentricity squared

# 모듈 전역 ENU 원점 (서버 기동 시 1회 설정)
_origin_lock = threading.Lock()
_origin_lla: Optional[Tuple[float, float, float]] = None
_origin_ecef: Optional[np.ndarray] = None
_ecef_to_enu_R: Optional[np.ndarray] = None


def _lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """위경도 → ECEF (Earth-Centered Earth-Fixed)"""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - _WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def _ecef_to_enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    """ECEF→ENU 회전 행렬 (원점 위경도 기준)"""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    return np.array([
        [-sin_lon,           cos_lon,           0.0    ],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat],
    ], dtype=np.float64)


def set_enu_origin(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> None:
    """
    로컬 ENU 좌표계의 원점을 설정 (서버 기동 시 1회 호출)

    Args:
        lat_deg: 원점 위도 (deg)
        lon_deg: 원점 경도 (deg)
        alt_m: 원점 고도 (m, ellipsoidal height)
    """
    global _origin_lla, _origin_ecef, _ecef_to_enu_R
    with _origin_lock:
        _origin_lla = (lat_deg, lon_deg, alt_m)
        _origin_ecef = _lla_to_ecef(lat_deg, lon_deg, alt_m)
        _ecef_to_enu_R = _ecef_to_enu_rotation(lat_deg, lon_deg)
        logger.info(
            f"ENU origin set: lat={lat_deg:.6f}, lon={lon_deg:.6f}, alt={alt_m:.2f}m"
        )


def geodetic_to_enu(
    lat_deg: float, lon_deg: float, alt_m: float
) -> np.ndarray:
    """
    위경도 → 로컬 ENU 좌표

    set_enu_origin() 호출 이후 사용 가능합니다. 미설정 시 첫 호출로 자동 원점 지정.

    Args:
        lat_deg, lon_deg, alt_m: 변환할 좌표

    Returns:
        np.ndarray (3,) [east, north, up] (m)
    """
    global _origin_ecef, _ecef_to_enu_R
    with _origin_lock:
        if _origin_ecef is None or _ecef_to_enu_R is None:
            # 자동 원점 설정 (최초 호출 좌표 사용)
            set_enu_origin(lat_deg, lon_deg, alt_m)
            return np.zeros(3, dtype=np.float64)

        ecef = _lla_to_ecef(lat_deg, lon_deg, alt_m)
        delta = ecef - _origin_ecef
        enu = _ecef_to_enu_R @ delta
        return enu


def get_enu_origin() -> Optional[Tuple[float, float, float]]:
    """현재 설정된 ENU 원점 (lat, lon, alt) 반환"""
    with _origin_lock:
        return _origin_lla
