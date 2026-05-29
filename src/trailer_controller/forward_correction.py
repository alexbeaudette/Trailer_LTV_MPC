from dataclasses import dataclass

import numpy as np

from .config import TrailerLtvMpcConfig
from .geometry import Measurement, measurement_from_explicit_state
from .math_utils import wrap_angle_difference
from .path_reference import PathReference
from .speed_profile import apply_start_end_speed_profile
from .trailer_ltv_mpc import ControllerCommand, ControllerOutput, TrailerLtvMpcController


@dataclass(frozen=True)
class CorrectionTarget:
    target_x_m: float
    target_y_m: float
    target_heading_rad: float
    anchor_x_m: float
    anchor_y_m: float


@dataclass
class ForwardCorrectionState:
    active: bool = False
    phase: str = "inactive"
    target: CorrectionTarget | None = None
    forward_path: PathReference | None = None
    reverse_path: PathReference | None = None
    forward_search_idx: int = 0


class ForwardCorrectionSupervisor:
    def __init__(self, controller: TrailerLtvMpcController, config: TrailerLtvMpcConfig | None = None):
        self.controller = controller
        self.config = config or controller.config
        self.state = ForwardCorrectionState()

    def start(self, measurement: Measurement, reverse_reference_path: PathReference, search_start_idx: int):
        target = compute_forward_correction_target(
            measurement, reverse_reference_path, search_start_idx, self.config
        )
        forward_path = build_locked_forward_path(target, measurement, self.config)
        reverse_path = build_locked_anchor_reverse_path(target, measurement)
        self.state = ForwardCorrectionState(True, "forward_to_target", target, forward_path, reverse_path, 0)

    def step(self, plant_state_explicit, u_prev, reverse_reference_path: PathReference, search_start_idx: int):
        measurement = measurement_from_explicit_state(plant_state_explicit, self.config.geom)
        if not self.state.active:
            self.start(measurement, reverse_reference_path, search_start_idx)

        if self.state.phase == "forward_to_target":
            output = pure_pursuit_forward_step(
                measurement, self.state.forward_path, self.state.forward_search_idx, self.config
            )
            self.state.forward_search_idx = output.search_start_idx
            if _forward_target_reached(measurement, self.state.target, self.config):
                self.state.phase = "reverse_to_anchor"
            return output

        output = self.controller.step(
            plant_state_explicit,
            u_prev,
            -abs(self.config.V2_reference_mps),
            self.state.reverse_path,
            0,
        )
        if np.hypot(measurement.X2 - self.state.target.anchor_x_m, measurement.Y2 - self.state.target.anchor_y_m) <= abs(
            self.config.forward_correction_anchor_reached_tolerance_m
        ):
            self.state = ForwardCorrectionState()
        return output


def pure_pursuit_forward_step(
    measurement: Measurement, forward_path: PathReference, search_idx: int, config: TrailerLtvMpcConfig
) -> ControllerOutput:
    lookahead_m = abs(config.forward_correction_forward_path_lookahead_m)
    projection = forward_path.project(measurement.X1, measurement.Y1, search_idx)
    target_station = min(projection.station_m + lookahead_m, forward_path.s_r[-1])
    sample = forward_path.sample(target_station)
    target_heading = float(np.arctan2(sample.y - measurement.Y1, sample.x - measurement.X1))
    lookahead_distance = max(float(np.hypot(sample.x - measurement.X1, sample.y - measurement.Y1)), np.finfo(float).eps)
    alpha = float(wrap_angle_difference(target_heading, measurement.psi1))
    delta_f_raw = float(np.arctan2(2.0 * config.geom.L1 * np.sin(alpha), lookahead_distance))
    delta_f = float(np.clip(delta_f_raw, -abs(config.delta_f_max_rad), abs(config.delta_f_max_rad)))
    profile = apply_start_end_speed_profile(
        abs(config.forward_correction_V1_mps), forward_path, config, projection.station_m
    )
    command = ControllerCommand(delta_f=delta_f, V1=profile.reference_profile, delta_T=np.nan, V2=np.nan)
    debug = {
        "mode": "forward_correction",
        "method": "pure_pursuit",
        "phase": "forward_to_target",
        "projection": projection,
        "lookahead_sample": sample,
        "target_heading_rad": target_heading,
        "alpha_rad": alpha,
        "delta_f_raw_rad": delta_f_raw,
        "speed_profile": profile,
    }
    return ControllerOutput(command, debug, projection.ref_idx)


def compute_forward_correction_target(
    measurement: Measurement, reference_path: PathReference, search_start_idx: int, config: TrailerLtvMpcConfig
) -> CorrectionTarget:
    projection = reference_path.project(measurement.X2, measurement.Y2, search_start_idx)
    target_station = min(
        projection.station_m + abs(config.forward_correction_target_distance_m), reference_path.s_r[-1]
    )
    sample = reference_path.sample(target_station)
    anchor = reference_path.sample(projection.station_m)
    return CorrectionTarget(sample.x, sample.y, sample.theta_ref, anchor.x, anchor.y)


def build_locked_forward_path(target: CorrectionTarget, measurement: Measurement, config: TrailerLtvMpcConfig):
    sample_spacing = 0.2
    target_xy = np.array([target.target_x_m, target.target_y_m], dtype=float)
    current_xy = np.array([measurement.X1, measurement.Y1], dtype=float)
    delta = target_xy - current_xy
    distance = max(float(np.linalg.norm(delta)), sample_spacing)
    if distance <= sample_spacing:
        unit = np.array([np.cos(target.target_heading_rad), np.sin(target.target_heading_rad)])
    else:
        unit = delta / distance
    offset = _path_start_offset(distance, sample_spacing, config)
    local_station = np.linspace(0.0, distance + offset, int(np.ceil((distance + offset) / sample_spacing)) + 1)
    line_station = local_station - offset
    xy = current_xy + np.outer(line_station, unit)
    return PathReference(
        xy[:, 0],
        xy[:, 1],
        np.full_like(local_station, np.arctan2(unit[1], unit[0])),
        local_station,
        np.ones_like(local_station),
        tracking_point="truck_rear_axle",
        type="forward_correction_forward_to_target",
    )


def build_locked_anchor_reverse_path(target: CorrectionTarget, measurement: Measurement):
    sample_spacing = 0.2
    current = np.array([measurement.X2, measurement.Y2])
    anchor = np.array([target.anchor_x_m, target.anchor_y_m])
    delta = anchor - current
    distance = max(float(np.linalg.norm(delta)), sample_spacing)
    unit = delta / distance if distance > sample_spacing else np.array([np.cos(target.target_heading_rad), np.sin(target.target_heading_rad)])
    station = np.linspace(0.0, distance, int(np.ceil(distance / sample_spacing)) + 1)
    xy = current + np.outer(station, unit)
    return PathReference(
        xy[:, 0],
        xy[:, 1],
        np.full_like(station, np.arctan2(unit[1], unit[0])),
        station,
        -np.ones_like(station),
        tracking_point="trailer_rear_axle",
        type="forward_correction_reverse_to_anchor",
    )


def _path_start_offset(target_station, sample_spacing, config):
    requested = config.forward_correction_path_start_offset_m
    if not np.isfinite(requested):
        requested = abs(config.V2_profile_start_ramp_m) * abs(
            config.forward_correction_path_start_offset_profile_fraction
        )
    return min(abs(requested), 0.25 * max(abs(target_station), sample_spacing))


def _forward_target_reached(measurement: Measurement, target: CorrectionTarget, config: TrailerLtvMpcConfig):
    error = np.array([measurement.X1 - target.target_x_m, measurement.Y1 - target.target_y_m])
    distance = float(np.linalg.norm(error))
    if distance <= abs(config.forward_correction_target_reached_tolerance_m):
        return True
    pass_radius = config.forward_correction_target_pass_radius_m
    if not np.isfinite(pass_radius):
        pass_radius = abs(config.V2_profile_end_ramp_m)
    heading_unit = np.array([np.cos(target.target_heading_rad), np.sin(target.target_heading_rad)])
    return distance <= abs(pass_radius) and float(error @ heading_unit) >= 0.0
