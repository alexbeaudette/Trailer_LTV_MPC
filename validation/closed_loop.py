"""Closed-loop validation runner for black-box controller checks."""

from dataclasses import dataclass

import numpy as np

from trailer_ltv_mpc import ForwardCorrectionSupervisor, TrailerLtvMpcConfig, TrailerLtvMpcController
from trailer_ltv_mpc.geometry import measurement_from_repo_state
from trailer_ltv_mpc.math_utils import wrap_angle_difference
from trailer_ltv_mpc.models import propagate_full_plant_one_step, recover_virtual_input_from_full_plant
from trailer_ltv_mpc.path_reference import PathReference


@dataclass(frozen=True)
class ClosedLoopResult:
    step_idx: np.ndarray
    t_s: np.ndarray
    repo_state: np.ndarray
    truck_rear_x: np.ndarray
    truck_rear_y: np.ndarray
    hitch_x: np.ndarray
    hitch_y: np.ndarray
    stations_m: np.ndarray
    errors_m: np.ndarray
    heading_errors_rad: np.ndarray
    gamma_rad: np.ndarray
    delta_f: np.ndarray
    V1: np.ndarray
    delta_T: np.ndarray
    delta_T_actual: np.ndarray
    V2: np.ndarray
    V2_actual: np.ndarray
    V2_ref: np.ndarray
    V2_profile_ref: np.ndarray
    V2_profile_start_scale: np.ndarray
    V2_profile_end_scale: np.ndarray
    V2_profile_combined_scale: np.ndarray
    solver_status: np.ndarray
    solver_iterations: np.ndarray
    search_start_idx: np.ndarray
    reference_x: np.ndarray
    reference_y: np.ndarray
    reference_theta: np.ndarray
    reference_s: np.ndarray
    reference_direction: np.ndarray
    mode: np.ndarray
    phase: np.ndarray
    correction_anchor_x: np.ndarray
    correction_anchor_y: np.ndarray
    correction_target_x: np.ndarray
    correction_target_y: np.ndarray
    metadata: dict

    @property
    def final_repo_state(self) -> np.ndarray:
        return self.repo_state[-1, :]

    @property
    def final_progress_m(self) -> float:
        return float(self.stations_m[-1] - self.stations_m[0])

    @property
    def max_error_m(self) -> float:
        return float(np.max(self.errors_m))

    @property
    def terminal_error_m(self) -> float:
        return float(self.errors_m[-1])

    @property
    def max_heading_error_rad(self) -> float:
        return float(np.max(np.abs(self.heading_errors_rad)))

    @property
    def max_gamma_rad(self) -> float:
        return float(np.max(np.abs(self.gamma_rad)))

    @property
    def path_length_m(self) -> float:
        return float(self.reference_s[-1] - self.reference_s[0])

    @property
    def final_progress_fraction(self) -> float:
        return float(self.final_progress_m / max(self.path_length_m, np.finfo(float).eps))

    @property
    def reached_end(self) -> bool:
        return bool(self.final_progress_m >= self.path_length_m - float(self.metadata.get("end_tolerance_m", 0.5)))

    @property
    def first_reached_end_idx(self) -> int | None:
        reached = np.flatnonzero(self.stations_m >= self.path_length_m - float(self.metadata.get("end_tolerance_m", 0.5)))
        return int(reached[0]) if reached.size else None

    @property
    def first_reached_end_time_s(self) -> float | None:
        idx = self.first_reached_end_idx
        return float(self.t_s[idx]) if idx is not None else None

    @property
    def solver_succeeded(self) -> bool:
        return bool(np.all(self.solver_status == "solved"))

    @property
    def speed_tracking_mask(self) -> np.ndarray:
        return (
            (self.V2_profile_combined_scale > 0.999)
            & (self.stations_m < self.path_length_m - float(self.metadata.get("end_tolerance_m", 0.5)))
            & (np.abs(self.V2_profile_ref) > float(self.metadata.get("speed_active_threshold_mps", 0.15)))
        )

    @property
    def max_speed_reference_error_mps(self) -> float:
        active = self.speed_tracking_mask
        if not np.any(active):
            return 0.0
        return float(np.max(np.abs(self.V2_actual[active] - self.V2_profile_ref[active])))

    @property
    def mean_speed_reference_error_mps(self) -> float:
        active = self.speed_tracking_mask
        if not np.any(active):
            return 0.0
        return float(np.mean(np.abs(self.V2_actual[active] - self.V2_profile_ref[active])))

    @property
    def mean_abs_cruise_V2_actual_mps(self) -> float:
        active = self.speed_tracking_mask
        if not np.any(active):
            return 0.0
        return float(np.mean(np.abs(self.V2_actual[active])))

    @property
    def mean_abs_cruise_V2_profile_ref_mps(self) -> float:
        active = self.speed_tracking_mask
        if not np.any(active):
            return 0.0
        return float(np.mean(np.abs(self.V2_profile_ref[active])))

    def summary(self) -> dict:
        return {
            "path_kind": self.metadata["path_kind"],
            "direction": self.metadata["direction"],
            "steps": int(self.metadata["steps"]),
            "ds": float(self.metadata["ds"]),
            "config_source": self.metadata["config_source"],
            "path_length_m": self.path_length_m,
            "final_progress_m": self.final_progress_m,
            "final_progress_fraction": self.final_progress_fraction,
            "reached_end": self.reached_end,
            "first_reached_end_time_s": self.first_reached_end_time_s,
            "max_error_m": self.max_error_m,
            "terminal_error_m": self.terminal_error_m,
            "max_heading_error_rad": self.max_heading_error_rad,
            "max_gamma_rad": self.max_gamma_rad,
            "solver_succeeded": self.solver_succeeded,
            "max_solver_iterations": int(np.max(self.solver_iterations)) if self.solver_iterations.size else 0,
            "mean_solver_iterations": float(np.mean(self.solver_iterations)) if self.solver_iterations.size else 0.0,
            "max_speed_reference_error_mps": self.max_speed_reference_error_mps,
            "mean_speed_reference_error_mps": self.mean_speed_reference_error_mps,
            "speed_tracking_sample_count": int(np.count_nonzero(self.speed_tracking_mask)),
            "mean_abs_cruise_V2_actual_mps": self.mean_abs_cruise_V2_actual_mps,
            "mean_abs_cruise_V2_profile_ref_mps": self.mean_abs_cruise_V2_profile_ref_mps,
            "mean_abs_V2_actual_mps": float(np.mean(np.abs(self.V2_actual))),
            "mean_abs_V2_profile_ref_mps": float(np.mean(np.abs(self.V2_profile_ref))),
            "reached_end_ok": self.reached_end,
            "lateral_error_ok": self.max_error_m <= float(self.metadata.get("lateral_error_tolerance_m", 0.75)),
            "heading_error_ok": self.max_heading_error_rad <= float(self.metadata.get("heading_error_tolerance_rad", 0.75)),
            "speed_reference_ok": self.max_speed_reference_error_mps
            <= float(self.metadata.get("speed_reference_tolerance_mps", 0.25)),
            "final_repo_state": self.final_repo_state.tolist(),
        }


def run_closed_loop_path(
    path: PathReference,
    direction_sign: float,
    config: TrailerLtvMpcConfig,
    steps: int = 120,
    metadata: dict | None = None,
) -> ClosedLoopResult:
    controller = TrailerLtvMpcController(config)
    forward_correction = ForwardCorrectionSupervisor(controller, config) if config.enable_forward_correction else None
    repo_state_current = initial_repo_state_from_path(path)
    u_prev = np.array([0.0, direction_sign * config.V2_min_abs_mps])
    V2_reference = direction_sign * abs(config.V2_reference_mps)
    search_start_idx = 0

    step_idx = np.arange(steps, dtype=int)
    t_s = step_idx * config.Ts
    repo_state = np.zeros((steps + 1, 4))
    repo_state[0, :] = repo_state_current
    truck_rear_x = np.zeros(steps)
    truck_rear_y = np.zeros(steps)
    hitch_x = np.zeros(steps)
    hitch_y = np.zeros(steps)
    stations = np.zeros(steps)
    errors = np.zeros(steps)
    heading_errors = np.zeros(steps)
    gamma = np.zeros(steps)
    delta_f = np.zeros(steps)
    V1 = np.zeros(steps)
    delta_T = np.zeros(steps)
    delta_T_actual = np.zeros(steps)
    V2 = np.zeros(steps)
    V2_actual = np.zeros(steps)
    V2_ref = V2_reference * np.ones(steps)
    V2_profile_ref = np.full(steps, np.nan)
    V2_profile_start_scale = np.full(steps, np.nan)
    V2_profile_end_scale = np.full(steps, np.nan)
    V2_profile_combined_scale = np.full(steps, np.nan)
    solver_status = np.full(steps, "not_run", dtype=object)
    solver_iterations = np.zeros(steps, dtype=int)
    search_indices = np.zeros(steps, dtype=int)
    mode = np.full(steps, "trailer_ltv_mpc", dtype=object)
    phase = np.full(steps, "tracking", dtype=object)
    correction_anchor_x = np.full(steps, np.nan)
    correction_anchor_y = np.full(steps, np.nan)
    correction_target_x = np.full(steps, np.nan)
    correction_target_y = np.full(steps, np.nan)

    for idx in range(steps):
        measurement = measurement_from_repo_state(repo_state_current, config.geom)
        explicit_state = measurement.explicit_state
        assert repo_state_current.shape == (4,)
        assert explicit_state.shape == (6,)
        assert u_prev.shape == (2,)

        projection = path.project(measurement.X2, measurement.Y2, search_start_idx)
        sample = path.sample(projection.station_m)
        stations[idx] = projection.station_m
        errors[idx] = projection.distance_m
        heading_errors[idx] = wrap_angle_difference(measurement.psi2, sample.theta_ref)
        gamma[idx] = measurement.gamma
        truck_rear_x[idx] = measurement.X1
        truck_rear_y[idx] = measurement.Y1
        hitch_x[idx] = measurement.Xh
        hitch_y[idx] = measurement.Yh
        search_indices[idx] = search_start_idx

        use_forward_correction = forward_correction is not None and (
            forward_correction.state.active
            or abs(measurement.gamma) >= abs(config.forward_correction_gamma_trigger_rad)
        )
        used_forward_correction = False
        if not use_forward_correction:
            try:
                output = controller.step(explicit_state, u_prev, V2_reference, path, search_start_idx)
            except ValueError as exc:
                if forward_correction is None or not _is_admissible_mapping_failure(exc):
                    raise
                used_forward_correction = True
                output = forward_correction.step(explicit_state, u_prev, path, search_start_idx)
        else:
            used_forward_correction = True
            output = forward_correction.step(explicit_state, u_prev, path, search_start_idx)
        command = output.command
        delta_f[idx] = command.delta_f
        V1[idx] = command.V1
        delta_T[idx] = command.delta_T
        V2[idx] = command.V2
        actual_virtual = recover_virtual_input_from_full_plant(
            repo_state_current, command.delta_f, command.V1, config.geom
        )
        delta_T_actual[idx] = actual_virtual.delta_T
        V2_actual[idx] = actual_virtual.V2
        mode[idx] = output.debug.get("mode", "trailer_ltv_mpc")
        phase[idx] = output.debug.get("phase", "tracking")
        if forward_correction is not None and forward_correction.state.target is not None:
            target = forward_correction.state.target
            correction_anchor_x[idx] = target.anchor_x_m
            correction_anchor_y[idx] = target.anchor_y_m
            correction_target_x[idx] = target.target_x_m
            correction_target_y[idx] = target.target_y_m
        profile_current = output.debug["ref"].V2_profile_current if "ref" in output.debug else output.debug["speed_profile"]
        V2_profile_ref[idx] = profile_current.reference_profile
        V2_profile_start_scale[idx] = profile_current.start_scale
        V2_profile_end_scale[idx] = profile_current.end_scale
        V2_profile_combined_scale[idx] = profile_current.combined_scale
        solver = output.debug.get("solver")
        if solver is not None:
            solver_status[idx] = solver.status
            solver_iterations[idx] = solver.iterations if solver.iterations is not None else 0
        else:
            solver_status[idx] = output.debug.get("method", "no_solver")

        repo_state_current = propagate_full_plant_one_step(
            repo_state_current, command.delta_f, command.V1, config.Ts, config.geom
        )
        assert repo_state_current.shape == (4,)
        repo_state[idx + 1, :] = repo_state_current
        if np.isfinite(command.delta_T) and np.isfinite(command.V2):
            u_prev = np.array([command.delta_T, command.V2])
        if used_forward_correction:
            search_start_idx = projection.ref_idx
        else:
            search_start_idx = output.search_start_idx

    result_metadata = {
        "path_kind": "",
        "direction": "",
        "steps": steps,
        "ds": np.nan,
        "config_source": "",
        "end_tolerance_m": 0.5,
        "lateral_error_tolerance_m": 0.75,
        "heading_error_tolerance_rad": 0.75,
        "speed_reference_tolerance_mps": 0.25,
        "speed_active_threshold_mps": 0.15,
    }
    if metadata:
        result_metadata.update(metadata)

    return ClosedLoopResult(
        step_idx=step_idx,
        t_s=t_s,
        repo_state=repo_state,
        truck_rear_x=truck_rear_x,
        truck_rear_y=truck_rear_y,
        hitch_x=hitch_x,
        hitch_y=hitch_y,
        stations_m=stations,
        errors_m=errors,
        heading_errors_rad=heading_errors,
        gamma_rad=gamma,
        delta_f=delta_f,
        V1=V1,
        delta_T=delta_T,
        delta_T_actual=delta_T_actual,
        V2=V2,
        V2_actual=V2_actual,
        V2_ref=V2_ref,
        V2_profile_ref=V2_profile_ref,
        V2_profile_start_scale=V2_profile_start_scale,
        V2_profile_end_scale=V2_profile_end_scale,
        V2_profile_combined_scale=V2_profile_combined_scale,
        solver_status=solver_status,
        solver_iterations=solver_iterations,
        search_start_idx=search_indices,
        reference_x=path.x_r.copy(),
        reference_y=path.y_r.copy(),
        reference_theta=path.theta_r.copy(),
        reference_s=path.s_r.copy(),
        reference_direction=path.dir_r.copy(),
        mode=mode,
        phase=phase,
        correction_anchor_x=correction_anchor_x,
        correction_anchor_y=correction_anchor_y,
        correction_target_x=correction_target_x,
        correction_target_y=correction_target_y,
        metadata=result_metadata,
    )


def initial_repo_state_from_path(path: PathReference) -> np.ndarray:
    sample = path.sample(path.s_r[0], path.delta_T_profile(1.0))
    psi2 = sample.theta_ref
    psi1 = psi2
    return np.array([sample.x, sample.y, psi1, psi2], dtype=float)


def assert_closed_loop_result(result: ClosedLoopResult, direction_sign: float, config: TrailerLtvMpcConfig):
    assert np.all(np.isfinite(result.final_repo_state))
    assert np.all(np.isfinite(result.repo_state))
    assert np.all(np.isfinite(result.truck_rear_x))
    assert np.all(np.isfinite(result.truck_rear_y))
    assert np.all(np.isfinite(result.hitch_x))
    assert np.all(np.isfinite(result.hitch_y))
    assert np.all(np.isfinite(result.stations_m))
    assert np.all(np.isfinite(result.errors_m))
    assert np.all(np.isfinite(result.heading_errors_rad))
    assert np.all(np.isfinite(result.gamma_rad))
    assert np.all(np.isfinite(result.delta_f))
    assert np.all(np.isfinite(result.V1))
    finite_virtual_cmd = np.isfinite(result.delta_T) & np.isfinite(result.V2)
    assert np.all(np.isfinite(result.delta_T[finite_virtual_cmd]))
    assert np.all(np.isfinite(result.V2[finite_virtual_cmd]))
    assert np.all(np.isfinite(result.delta_T_actual))
    assert np.all(np.isfinite(result.V2_actual))
    assert np.all(np.abs(result.delta_f) <= config.delta_f_max_rad + 1.0e-9)

    active_speed = finite_virtual_cmd & (np.abs(result.V2) > 1.0e-3)
    assert np.all(np.sign(result.V2[active_speed]) == direction_sign)
    assert np.all(np.diff(result.stations_m) >= -1.0e-9)
    assert result.final_progress_m > 1.0
    assert result.max_error_m < 0.75
    assert result.terminal_error_m < 0.75
    assert result.max_heading_error_rad < 0.75
    assert result.max_gamma_rad < config.forward_correction_gamma_trigger_rad


def _is_admissible_mapping_failure(error: ValueError) -> bool:
    message = str(error)
    return (
        "No admissible delta_T interval exists" in message
        or "Admissible delta_T interval collapsed" in message
        or "Mapping denominator is too small" in message
    )
