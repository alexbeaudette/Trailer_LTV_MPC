from dataclasses import dataclass

import numpy as np

from .math_utils import wrap_to_pi


@dataclass(frozen=True)
class Projection:
    distance_m: float
    station_m: float
    segment_idx: int
    segment_fraction: float
    ref_idx: int


@dataclass(frozen=True)
class PathSample:
    x: float
    y: float
    theta_ref: float
    delta_T_ref: float
    direction: float
    station_m: float
    ref_idx: int


@dataclass
class PathReference:
    x_r: np.ndarray
    y_r: np.ndarray
    theta_r: np.ndarray
    s_r: np.ndarray
    dir_r: np.ndarray
    tracking_point: str = "trailer_rear_axle"
    type: str = ""

    def __post_init__(self):
        self.x_r = np.asarray(self.x_r, dtype=float).reshape(-1)
        self.y_r = np.asarray(self.y_r, dtype=float).reshape(-1)
        self.theta_r = np.asarray(self.theta_r, dtype=float).reshape(-1)
        self.s_r = np.asarray(self.s_r, dtype=float).reshape(-1)
        self.dir_r = np.asarray(self.dir_r, dtype=float).reshape(-1)
        n = len(self.s_r)
        if not (len(self.x_r) == len(self.y_r) == len(self.theta_r) == len(self.dir_r) == n):
            raise ValueError("PathReference arrays must have matching lengths.")
        if n < 2:
            raise ValueError("PathReference requires at least two samples.")
        if np.any(np.diff(self.s_r) <= 0.0):
            raise ValueError("PathReference.s_r must be strictly increasing.")
        self.tracking_point = _normalize_tracking_point(self.tracking_point)

    @property
    def length_m(self) -> float:
        return float(self.s_r[-1])

    def delta_T_profile(self, trailer_wheelbase_m: float) -> np.ndarray:
        theta = wrap_to_pi(self.theta_r)
        kappa = np.zeros_like(theta)
        for idx in range(len(theta) - 1):
            ds = self.s_r[idx + 1] - self.s_r[idx]
            dtheta = wrap_to_pi(theta[idx + 1] - theta[idx])
            kappa[idx] = dtheta / ds
        kappa[-1] = kappa[-2]
        direction = np.where(self.dir_r == 0.0, 1.0, self.dir_r)
        return np.arctan(direction * trailer_wheelbase_m * kappa)

    def project(self, x_point: float, y_point: float, start_idx: int = 0) -> Projection:
        n = len(self.s_r)
        search_start = int(np.clip(start_idx, 0, n - 2))
        best_distance = np.inf
        best_station = float(self.s_r[search_start])
        best_idx = search_start
        best_t = 0.0
        for idx in range(search_start, n - 1):
            x0 = self.x_r[idx]
            y0 = self.y_r[idx]
            dx = self.x_r[idx + 1] - x0
            dy = self.y_r[idx + 1] - y0
            length_sq = dx * dx + dy * dy
            if length_sq <= np.finfo(float).eps:
                t = 0.0
            else:
                t = ((x_point - x0) * dx + (y_point - y0) * dy) / length_sq
                t = float(np.clip(t, 0.0, 1.0))
            x_proj = x0 + t * dx
            y_proj = y0 + t * dy
            distance = float(np.hypot(x_proj - x_point, y_proj - y_point))
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
                best_t = t
                best_station = float(self.s_r[idx] + t * (self.s_r[idx + 1] - self.s_r[idx]))
        return Projection(best_distance, best_station, best_idx, best_t, self.station_to_index(best_station))

    def sample(self, station_query: float, delta_T_profile=None) -> PathSample:
        station = float(np.clip(station_query, self.s_r[0], self.s_r[-1]))
        theta_unwrapped = np.unwrap(self.theta_r)
        delta_T = self.delta_T_profile(1.0) if delta_T_profile is None else np.asarray(delta_T_profile)
        direction = float(np.interp(station, self.s_r, self.dir_r))
        if not np.isfinite(direction) or direction == 0.0:
            direction = float(self.dir_r[self.station_to_index(station)])
        direction = float(np.sign(direction) or 1.0)
        return PathSample(
            x=float(np.interp(station, self.s_r, self.x_r)),
            y=float(np.interp(station, self.s_r, self.y_r)),
            theta_ref=float(wrap_to_pi(np.interp(station, self.s_r, theta_unwrapped))),
            delta_T_ref=float(np.interp(station, self.s_r, delta_T)),
            direction=direction,
            station_m=station,
            ref_idx=self.station_to_index(station),
        )

    def station_to_index(self, station_query: float) -> int:
        station = float(np.clip(station_query, self.s_r[0], self.s_r[-1]))
        return int(np.argmin(np.abs(self.s_r - station)))


def _normalize_tracking_point(value: str) -> str:
    text = str(value).lower().strip().replace("-", "_").replace(" ", "_")
    if text == "hitch":
        return "hitch"
    if text == "truck_rear_axle":
        return "truck_rear_axle"
    return "trailer_rear_axle"
