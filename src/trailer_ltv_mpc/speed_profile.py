from dataclasses import dataclass

import numpy as np

from .config import TrailerLtvMpcConfig
from .geometry import Measurement
from .math_utils import clamp, ramp_score, wrap_angle_difference
from .path_reference import PathReference


@dataclass(frozen=True)
class SpeedProfile:
    enabled: bool
    active: bool
    station_m: float
    remaining_m: float
    start_scale: float
    end_scale: float
    combined_scale: float
    reference_in: float
    reference_profile: float


def apply_start_end_speed_profile(
    V_reference: float, path: PathReference, config: TrailerLtvMpcConfig, station_m: float
) -> SpeedProfile:
    enabled = bool(config.enable_start_end_V2_profile)
    station = float(clamp(station_m, path.s_r[0], path.s_r[-1]))
    remaining = max(path.s_r[-1] - station, 0.0)
    start_scale = 1.0
    end_scale = 1.0
    if enabled:
        if config.V2_profile_start_ramp_m > 0.0:
            start_scale = float(clamp(station / abs(config.V2_profile_start_ramp_m), 0.0, 1.0))
        if config.V2_profile_end_ramp_m > 0.0:
            end_scale = float(clamp(remaining / abs(config.V2_profile_end_ramp_m), 0.0, 1.0))
    combined = min(start_scale, end_scale)
    return SpeedProfile(
        enabled=enabled,
        active=enabled and combined < 0.999,
        station_m=station,
        remaining_m=float(remaining),
        start_scale=start_scale,
        end_scale=end_scale,
        combined_scale=combined,
        reference_in=float(V_reference),
        reference_profile=float(V_reference * combined),
    )


def compute_tracking_diagnostic(
    X_mpc,
    measurement: Measurement,
    path: PathReference,
    search_start_idx: int,
):
    if path.tracking_point == "hitch":
        x = measurement.Xh
        y = measurement.Yh
    else:
        x = float(X_mpc[0])
        y = float(X_mpc[1])
    projection = path.project(x, y, search_start_idx)
    sample = path.sample(projection.station_m)
    dx = x - sample.x
    dy = y - sample.y
    return {
        "ref_idx": projection.ref_idx,
        "tracking_point": path.tracking_point,
        "cross_track_error_m": float(-np.sin(sample.theta_ref) * dx + np.cos(sample.theta_ref) * dy),
        "heading_error_rad": float(wrap_angle_difference(float(X_mpc[2]), sample.theta_ref)),
    }


def compute_effective_V2_reference(
    X_mpc,
    measurement: Measurement,
    V2_reference_nominal: float,
    path: PathReference,
    config: TrailerLtvMpcConfig,
    search_start_idx: int,
):
    tracking = compute_tracking_diagnostic(X_mpc, measurement, path, search_start_idx)
    adaptive = {
        "enabled": False,
        "nominal_reference": float(V2_reference_nominal),
        "effective_reference": float(V2_reference_nominal),
        "speed_scale": 1.0,
        "trigger": "disabled",
        **tracking,
        "gamma_rad": measurement.gamma,
        "cte_score": 0.0,
        "heading_score": 0.0,
        "gamma_score": 0.0,
    }
    if not config.enable_adaptive_V2_reference:
        return float(V2_reference_nominal), adaptive

    adaptive["enabled"] = True
    adaptive["trigger"] = "nominal"
    min_scale = float(clamp(config.adaptive_V2_min_scale, 0.0, 1.0))
    adaptive["cte_score"] = ramp_score(
        abs(tracking["cross_track_error_m"]),
        config.adaptive_V2_cte_start_m,
        config.adaptive_V2_cte_full_m,
    )
    adaptive["heading_score"] = ramp_score(
        abs(tracking["heading_error_rad"]),
        config.adaptive_V2_heading_start_rad,
        config.adaptive_V2_heading_full_rad,
    )
    adaptive["gamma_score"] = ramp_score(
        abs(measurement.gamma),
        config.adaptive_V2_gamma_start_rad,
        config.adaptive_V2_gamma_full_rad,
    )
    scores = np.array([adaptive["cte_score"], adaptive["heading_score"], adaptive["gamma_score"]])
    dominant_idx = int(np.argmax(scores))
    if scores[dominant_idx] > 0.0:
        adaptive["trigger"] = ["cross_track", "heading", "articulation"][dominant_idx]
    speed_scale = 1.0 - (1.0 - min_scale) * float(scores[dominant_idx])
    V2_effective = apply_speed_scale_with_bounds(V2_reference_nominal, speed_scale, config)
    adaptive["effective_reference"] = V2_effective
    adaptive["speed_scale"] = abs(V2_effective) / max(abs(V2_reference_nominal), np.finfo(float).eps)
    return V2_effective, adaptive


def apply_speed_scale_with_bounds(V2_reference: float, speed_scale: float, config: TrailerLtvMpcConfig):
    direction = float(np.sign(V2_reference) or 1.0)
    scaled_abs = abs(V2_reference) * speed_scale
    scaled_abs = float(clamp(scaled_abs, abs(config.V2_min_abs_mps), abs(config.V2_max_abs_mps)))
    return direction * scaled_abs
