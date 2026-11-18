"""
Geographic and spatial calculation utilities for trajectory map-matching.

Includes Haversine distance, bearings, angular differences, local Cartesian projections,
and orthogonal point-to-segment projections.
"""

from typing import Tuple, Union
import math
import numpy as np

# Standard WGS84 constants
EARTH_RADIUS_METERS = 6371000.0


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Computes the great-circle distance between two points in meters using the Haversine formula.

    Parameters
    ----------
    lon1, lat1 : float
        Coordinates of first point (degrees).
    lon2, lat2 : float
        Coordinates of second point (degrees).

    Returns
    -------
    float
        Distance in meters.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def haversine_distance_vectorized(
    lons1: np.ndarray,
    lats1: np.ndarray,
    lons2: np.ndarray,
    lats2: np.ndarray
) -> np.ndarray:
    """
    Vectorized Haversine distance calculation in meters.
    Broadcasts across inputs of matching or broadcastable shape.
    """
    phi1 = np.radians(lats1)
    phi2 = np.radians(lats2)
    delta_phi = np.radians(lats2 - lats1)
    delta_lambda = np.radians(lons2 - lons1)

    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2))
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def initial_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Computes the initial bearing (forward azimuth) from point 1 to point 2 in degrees [0, 360).

    Returns
    -------
    float
        Bearing in degrees, clockwise from true North (0 = North, 90 = East, 180 = South, 270 = West).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2) -
         math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda))
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def initial_bearing_vectorized(
    lons1: np.ndarray,
    lats1: np.ndarray,
    lons2: np.ndarray,
    lats2: np.ndarray
) -> np.ndarray:
    """
    Vectorized calculation of initial bearing in degrees [0, 360).
    """
    phi1 = np.radians(lats1)
    phi2 = np.radians(lats2)
    delta_lambda = np.radians(lons2 - lons1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = (np.cos(phi1) * np.sin(phi2) -
         np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda))
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360.0) % 360.0


def bearing_difference(bearing1: Union[float, np.ndarray], bearing2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the minimal unsigned angular difference between two bearings in degrees [0, 180].

    Parameters
    ----------
    bearing1, bearing2 : float or np.ndarray
        Bearings in degrees [0, 360).

    Returns
    -------
    float or np.ndarray
        Difference in degrees within [0, 180].
    """
    diff = np.abs(np.asarray(bearing1) - np.asarray(bearing2)) % 360.0
    res = np.where(diff > 180.0, 360.0 - diff, diff)
    return float(res) if np.ndim(bearing1) == 0 and np.ndim(bearing2) == 0 else res


def local_cartesian_projection(
    lons: np.ndarray,
    lats: np.ndarray,
    ref_lat: float = None,
    ref_lon: float = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Projects geographic coordinates (lon, lat) to local Cartesian metric plane (x, y in meters)
    centered on a reference coordinate using equirectangular projection with latitude correction.

    Preserves Euclidean distance with sub-meter accuracy within city/regional scale (~100km radius).

    Returns
    -------
    x_m : np.ndarray
        X coordinates in meters (East).
    y_m : np.ndarray
        Y coordinates in meters (North).
    kx : float
        Meters per degree longitude at reference latitude.
    ky : float
        Meters per degree latitude.
    """
    if ref_lat is None:
        ref_lat = float(np.mean(lats))
    if ref_lon is None:
        ref_lon = float(np.mean(lons))

    # WGS84 meridian and parallel scale factors
    rad_lat = math.radians(ref_lat)
    kx = 111320.0 * math.cos(rad_lat)
    ky = 110540.0

    x_m = (lons - ref_lon) * kx
    y_m = (lats - ref_lat) * ky
    return x_m, y_m, kx, ky


def point_to_segment_projection(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float
) -> Tuple[float, float, float, float]:
    """
    Projects a 2D Cartesian point P(px, py) onto directed line segment A(ax, ay) -> B(bx, by).

    Parameters
    ----------
    px, py : float
        Query point coordinates.
    ax, ay : float
        Segment start vertex coordinates.
    bx, by : float
        Segment end vertex coordinates.

    Returns
    -------
    perp_dist : float
        Orthogonal distance in meters from point P to the closest point on segment AB.
    t_proj : float
        Normalized projection fraction along segment AB, clamped to [0.0, 1.0].
        0.0 = point projects to or beyond vertex A; 1.0 = point projects to or beyond vertex B.
    proj_x, proj_y : float
        Coordinates of the projection point on the segment.
    """
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy

    if l2 <= 1e-12:
        # Degenerate zero-length segment
        dist = math.hypot(px - ax, py - ay)
        return dist, 0.0, ax, ay

    # Normalized projection scalar
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    t_clamped = max(0.0, min(1.0, t))

    proj_x = ax + t_clamped * dx
    proj_y = ay + t_clamped * dy
    perp_dist = math.hypot(px - proj_x, py - proj_y)

    return perp_dist, t_clamped, proj_x, proj_y


def batch_point_to_segment_projection(
    px: float,
    py: float,
    seg_ax: np.ndarray,
    seg_ay: np.ndarray,
    seg_bx: np.ndarray,
    seg_by: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized projection of a single point P(px, py) onto an array of segments A->B.

    Parameters
    ----------
    px, py : float
        Query point coordinates in meters.
    seg_ax, seg_ay : np.ndarray
        Start coordinates (N,) of segments.
    seg_bx, seg_by : np.ndarray
        End coordinates (N,) of segments.

    Returns
    -------
    dists : np.ndarray (N,)
        Orthogonal distances from P to each segment.
    t_proj : np.ndarray (N,)
        Clamped projection fractions [0.0, 1.0].
    proj_x, proj_y : np.ndarray (N,)
        Projected point coordinates on segments.
    """
    dx = seg_bx - seg_ax
    dy = seg_by - seg_ay
    l2 = dx * dx + dy * dy

    # Guard against zero-length segments
    valid_l2 = np.where(l2 <= 1e-12, 1.0, l2)
    t = ((px - seg_ax) * dx + (py - seg_ay) * dy) / valid_l2
    t_clamped = np.clip(t, 0.0, 1.0)
    t_clamped = np.where(l2 <= 1e-12, 0.0, t_clamped)

    proj_x = seg_ax + t_clamped * dx
    proj_y = seg_ay + t_clamped * dy

    dists = np.hypot(px - proj_x, py - proj_y)
    return dists, t_clamped, proj_x, proj_y
