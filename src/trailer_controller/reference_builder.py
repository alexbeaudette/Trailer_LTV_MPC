from dataclasses import dataclass

import numpy as np

from .config import TrailerLtvMpcConfig
from .math_utils import wrap_angle_difference
from .models import propagate_trailer_virtual_state_one_step
from .path_reference import PathReference
from .speed_profile import SpeedProfile, apply_start_end_speed_profile


@dataclass(frozen=True)
class V2ProfileHorizon:
    enabled: np.ndarray
    active: np.ndarray
    station_m: np.ndarray
    remaining_m: np.ndarray
    start_scale: np.ndarray
    end_scale: np.ndarray
    combined_scale: np.ndarray
    reference_in: np.ndarray
    reference_profile: np.ndarray


@dataclass(frozen=True)
class ReferenceBundle:
    errors: float
    X_ref0: float
    Y_ref0: float
    psi2_ref0: float
    delta_T_ref0: float
    i0: int
    s0: float
    indices: np.ndarray
    X_ref: np.ndarray
    Y_ref: np.ndarray
    delta_T_ref: np.ndarray
    V2_ref: np.ndarray
    U_ref: np.ndarray
    X_lin: np.ndarray
    delta_T_lin: np.ndarray
    V2_lin: np.ndarray
    U_lin: np.ndarray
    V2_profile_current: SpeedProfile
    V2_profile: V2ProfileHorizon
    output_tracking_point: str
    use_output_tracking: bool
    ref_mode: float


def generate_trailer_ltv_mpc_reference(
    X0,
    u_prev,
    V2_reference: float,
    path: PathReference,
    config: TrailerLtvMpcConfig,
    search_start_idx: int = 0,
) -> ReferenceBundle:
    X0 = np.asarray(X0, dtype=float).reshape(3)
    u_prev = np.asarray(u_prev, dtype=float).reshape(2)
    delta_T_profile = path.delta_T_profile(config.geom.L2)

    tracking_x0, tracking_y0 = _tracking_point_xy(X0, path.tracking_point, config)
    projection = path.project(tracking_x0, tracking_y0, search_start_idx)
    current = path.sample(projection.station_m, delta_T_profile)
    psi2_ref0 = _align_heading_reference(current.theta_ref, X0[2])
    V2_profile_current = apply_start_end_speed_profile(V2_reference, path, config, projection.station_m)

    X_ref, Y_ref, delta_T_ref, V2_ref, indices, V2_profile = _predictive_horizon(
        X0, u_prev, V2_reference, path, delta_T_profile, config, projection
    )
    X_lin, delta_T_lin, V2_lin = _constant_current_linearization_reference(X0, u_prev, V2_ref, config)
    U_ref = np.vstack([delta_T_ref, V2_ref])
    U_lin = np.vstack([delta_T_lin, V2_lin])
    ref_mode = current.direction if current.direction != 0.0 else 1.0

    return ReferenceBundle(
        errors=projection.distance_m,
        X_ref0=current.x,
        Y_ref0=current.y,
        psi2_ref0=psi2_ref0,
        delta_T_ref0=current.delta_T_ref,
        i0=projection.ref_idx,
        s0=projection.station_m,
        indices=indices,
        X_ref=X_ref,
        Y_ref=Y_ref,
        delta_T_ref=delta_T_ref,
        V2_ref=V2_ref,
        U_ref=U_ref,
        X_lin=X_lin,
        delta_T_lin=delta_T_lin,
        V2_lin=V2_lin,
        U_lin=U_lin,
        V2_profile_current=V2_profile_current,
        V2_profile=V2_profile,
        output_tracking_point=path.tracking_point,
        use_output_tracking=path.tracking_point == "hitch",
        ref_mode=ref_mode,
    )


def _predictive_horizon(X0, u_prev, V2_reference, path, delta_T_profile, config, projection):
    Xk = np.asarray(X0, dtype=float).reshape(3).copy()
    delta_T_prev = float(u_prev[0])
    V2_prev = float(u_prev[1])
    X_ref = np.zeros((config.nx, config.N))
    Y_ref = np.zeros((config.nx, config.N))
    delta_T_ref = np.zeros(config.N)
    V2_ref = np.zeros(config.N)
    indices = np.zeros(config.N, dtype=int)
    profile_entries = []
    search_start = projection.ref_idx
    station_start = projection.station_m
    use_station_preview = bool(config.enable_start_end_V2_profile)
    station_step = abs(V2_reference) * config.Ts

    for stage_idx in range(config.N):
        Xk = propagate_trailer_virtual_state_one_step(
            Xk, delta_T_prev, V2_prev, config.Ts, config.geom
        )
        if use_station_preview:
            station_target = min(station_start + (stage_idx + 1) * station_step, path.s_r[-1])
            sample = path.sample(station_target, delta_T_profile)
            search_start = sample.ref_idx
        else:
            tracking_x, tracking_y = _tracking_point_xy(Xk, path.tracking_point, config)
            stage_projection = path.project(tracking_x, tracking_y, search_start)
            sample = path.sample(stage_projection.station_m, delta_T_profile)
            search_start = sample.ref_idx

        psi2_ref = _align_heading_reference(sample.theta_ref, Xk[2])
        Y_ref[:, stage_idx] = [sample.x, sample.y, psi2_ref]
        X_ref[:, stage_idx] = [
            _tracking_to_state_x(sample.x, sample.theta_ref, path.tracking_point, config),
            _tracking_to_state_y(sample.y, sample.theta_ref, path.tracking_point, config),
            psi2_ref,
        ]
        delta_T_ref[stage_idx] = sample.delta_T_ref
        profile = apply_start_end_speed_profile(V2_reference, path, config, sample.station_m)
        V2_ref[stage_idx] = profile.reference_profile
        profile_entries.append(profile)
        indices[stage_idx] = search_start

    return X_ref, Y_ref, delta_T_ref, V2_ref, indices, _pack_profile(profile_entries)


def _constant_current_linearization_reference(X0, u_prev, V2_ref, config):
    delta_T_prev = float(u_prev[0])
    V2_prev = float(u_prev[1])
    threshold = abs(config.V2_profile_stop_tolerance_mps)
    V2_linearization = V2_prev if abs(V2_prev) >= threshold else float(V2_ref[0])
    X_lin = np.repeat(np.asarray(X0, dtype=float).reshape(3, 1), config.N, axis=1)
    delta_T_lin = delta_T_prev * np.ones(config.N)
    V2_lin = V2_linearization * np.ones(config.N)
    return X_lin, delta_T_lin, V2_lin


def _pack_profile(entries):
    return V2ProfileHorizon(
        enabled=np.array([p.enabled for p in entries], dtype=bool),
        active=np.array([p.active for p in entries], dtype=bool),
        station_m=np.array([p.station_m for p in entries], dtype=float),
        remaining_m=np.array([p.remaining_m for p in entries], dtype=float),
        start_scale=np.array([p.start_scale for p in entries], dtype=float),
        end_scale=np.array([p.end_scale for p in entries], dtype=float),
        combined_scale=np.array([p.combined_scale for p in entries], dtype=float),
        reference_in=np.array([p.reference_in for p in entries], dtype=float),
        reference_profile=np.array([p.reference_profile for p in entries], dtype=float),
    )


def _tracking_point_xy(X, tracking_point, config):
    if tracking_point == "hitch":
        return (
            float(X[0] + config.geom.L2 * np.cos(X[2])),
            float(X[1] + config.geom.L2 * np.sin(X[2])),
        )
    return float(X[0]), float(X[1])


def _tracking_to_state_x(x_ref, theta_ref, tracking_point, config):
    if tracking_point == "hitch":
        return float(x_ref - config.geom.L2 * np.cos(theta_ref))
    return float(x_ref)


def _tracking_to_state_y(y_ref, theta_ref, tracking_point, config):
    if tracking_point == "hitch":
        return float(y_ref - config.geom.L2 * np.sin(theta_ref))
    return float(y_ref)


def _align_heading_reference(psi2_ref, psi2_state):
    return float(psi2_state + wrap_angle_difference(psi2_ref, psi2_state))
