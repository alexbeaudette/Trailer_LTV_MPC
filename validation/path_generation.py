"""Deterministic path generators for validation runs."""

import numpy as np
from scipy.interpolate import CubicSpline

from trailer_ltv_mpc.path_reference import PathReference


def make_validation_path(kind: str, direction: str, ds: float = 0.2) -> PathReference:
    sign = direction_sign(direction)
    if kind == "straight":
        return _straight_path(sign, ds)
    if kind == "arc":
        return _arc_path(sign, ds)
    if kind == "spline":
        return _spline_path(sign, ds)
    if kind == "harsh_turn":
        return _harsh_turn_path(sign, ds)
    raise ValueError(f"Unknown validation path kind: {kind}.")


def direction_sign(direction: str) -> float:
    text = str(direction).lower().strip()
    if text == "forward":
        return 1.0
    if text == "reverse":
        return -1.0
    raise ValueError(f"Unknown direction: {direction}.")


def _straight_path(sign: float, ds: float) -> PathReference:
    length_m = 24.0
    station = np.arange(0.0, length_m + ds, ds)
    x = station.copy()
    y = np.zeros_like(station)
    tangent = np.zeros_like(station)
    return _path_from_tangent(x, y, station, tangent, sign)


def _arc_path(sign: float, ds: float) -> PathReference:
    radius_m = 18.0
    angle_rad = np.pi / 3.0
    length_m = radius_m * angle_rad
    station = np.arange(0.0, length_m + ds, ds)
    phi = station / radius_m
    x = radius_m * np.sin(phi)
    y = radius_m * (1.0 - np.cos(phi))
    tangent = phi
    return _path_from_tangent(x, y, station, tangent, sign)


def _spline_path(sign: float, ds: float) -> PathReference:
    waypoints = np.array(
        [
            [0.0, 0.0],
            [16.0, 0.0],
            [32.0, 7.0],
            [55.0, 9.0],
            [78.0, 0.0],
            [94.0, -4.0],
            [114.0, 0.0],
        ],
        dtype=float,
    )
    waypoint_station = _station_from_xy(waypoints[:, 0], waypoints[:, 1])
    sample_station = _sample_station(waypoint_station[-1], ds)
    x = CubicSpline(waypoint_station, waypoints[:, 0])(sample_station)
    y = CubicSpline(waypoint_station, waypoints[:, 1])(sample_station)
    station = _station_from_xy(x, y)
    tangent = _tangent_from_xy(x, y)
    return _path_from_tangent(x, y, station, tangent, sign)


def _harsh_turn_path(sign: float, ds: float) -> PathReference:
    approach_length = 25.0
    corner_size = 7.0
    exit_length = 25.0
    turn_angle = np.deg2rad(88.0)
    turn_sign = 1.0
    handle_length = 0.65 * corner_size

    approach_s = np.arange(0.0, approach_length + ds, ds)
    approach = np.column_stack([approach_s, np.zeros_like(approach_s)])

    theta_out = turn_sign * turn_angle
    p0 = np.array([approach_length, 0.0])
    p3 = p0 + corner_size * np.array([np.cos(0.5 * theta_out), np.sin(0.5 * theta_out)])
    t0 = np.array([1.0, 0.0])
    t1 = np.array([np.cos(theta_out), np.sin(theta_out)])
    p1 = p0 + handle_length * t0
    p2 = p3 - handle_length * t1

    tau = np.linspace(0.0, 1.0, max(25, int(np.ceil(4.0 * corner_size / ds))))
    corner = _cubic_bezier(p0, p1, p2, p3, tau)
    exit_s = _sample_station(exit_length, ds)[1:]
    exit_segment = p3 + np.outer(exit_s, t1)

    points = np.vstack([approach, corner[1:, :], exit_segment])
    sampled = _resample_xy(points[:, 0], points[:, 1], ds)
    x = sampled[:, 0]
    y = sampled[:, 1]
    station = _station_from_xy(x, y)
    tangent = _tangent_from_xy(x, y)
    return _path_from_tangent(x, y, station, tangent, sign)


def _path_from_tangent(x, y, station, tangent, sign: float) -> PathReference:
    heading = np.asarray(tangent, dtype=float)
    if sign < 0.0:
        heading = heading + np.pi
    return PathReference(
        x_r=np.asarray(x, dtype=float),
        y_r=np.asarray(y, dtype=float),
        theta_r=heading,
        s_r=np.asarray(station, dtype=float),
        dir_r=sign * np.ones_like(station, dtype=float),
    )


def _cubic_bezier(p0, p1, p2, p3, tau) -> np.ndarray:
    one_minus = 1.0 - tau.reshape(-1, 1)
    tau = tau.reshape(-1, 1)
    return one_minus**3 * p0 + 3.0 * one_minus**2 * tau * p1 + 3.0 * one_minus * tau**2 * p2 + tau**3 * p3


def _resample_xy(x, y, ds: float) -> np.ndarray:
    station = _station_from_xy(x, y)
    keep = np.concatenate([[True], np.diff(station) > 1.0e-12])
    station = station[keep]
    x = np.asarray(x, dtype=float)[keep]
    y = np.asarray(y, dtype=float)[keep]
    query_station = _sample_station(station[-1], ds)
    return np.column_stack(
        [
            np.interp(query_station, station, x),
            np.interp(query_station, station, y),
        ]
    )


def _station_from_xy(x, y) -> np.ndarray:
    segment = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(segment)])


def _sample_station(length_m: float, ds: float) -> np.ndarray:
    station = np.arange(0.0, length_m + ds, ds)
    if abs(station[-1] - length_m) > 1.0e-12:
        station = np.concatenate([station[station < length_m], [length_m]])
    return station


def _tangent_from_xy(x, y) -> np.ndarray:
    dx = np.diff(np.asarray(x, dtype=float))
    dy = np.diff(np.asarray(y, dtype=float))
    tangent = np.arctan2(dy, dx)
    return np.concatenate([tangent, [tangent[-1]]])
