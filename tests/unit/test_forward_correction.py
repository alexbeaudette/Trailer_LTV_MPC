import numpy as np
import pytest

from trailer_ltv_mpc.config import TrailerLtvMpcConfig

from trailer_ltv_mpc.forward_correction import (
    CorrectionTarget,
    ForwardCorrectionState,
    ForwardCorrectionSupervisor,
    build_locked_anchor_reverse_path,
    build_locked_forward_path,
    compute_forward_correction_target,
    pure_pursuit_forward_step,
)
from trailer_ltv_mpc.geometry import measurement_from_repo_state
from trailer_ltv_mpc.trailer_ltv_mpc import ControllerCommand, ControllerOutput
from validation.path_generation import make_validation_path


def test_pure_pursuit_forward_correction_outputs_finite_command():
    config = TrailerLtvMpcConfig()
    measurement = measurement_from_repo_state([0.0, 0.0, 0.0, 0.0], config.geom)
    target = CorrectionTarget(15.0, 0.0, 0.0, 0.0, 0.0)
    path = build_locked_forward_path(target, measurement, config)
    output = pure_pursuit_forward_step(measurement, path, 0, config)
    assert np.isfinite(output.command.delta_f)
    assert np.isfinite(output.command.V1)
    assert abs(output.command.delta_f) <= config.delta_f_max_rad


def test_locked_anchor_reverse_path_uses_reverse_heading_convention():
    config = TrailerLtvMpcConfig()
    measurement = measurement_from_repo_state([0.0, 0.0, 0.0, 0.0], config.geom)
    target = CorrectionTarget(15.0, 0.0, 0.0, 0.0, 0.0)

    path = build_locked_anchor_reverse_path(target, measurement)
    dx = np.diff(path.x_r)
    dy = np.diff(path.y_r)
    segment_length = np.hypot(dx, dy)
    tangent_unit = np.column_stack([dx / segment_length, dy / segment_length])
    heading_unit = np.column_stack([np.cos(path.theta_r[:-1]), np.sin(path.theta_r[:-1])])

    assert np.all(path.dir_r == -1.0)
    assert np.all(np.sum(tangent_unit * heading_unit, axis=1) < -0.999)


def test_forward_correction_target_projects_from_anchor_heading():
    config = TrailerLtvMpcConfig(forward_correction_target_distance_m=50.0)
    path = make_validation_path("harsh_turn", "reverse")
    measurement = measurement_from_repo_state(
        [28.0, 0.15, np.deg2rad(-45.0), path.sample(28.0).theta_ref],
        config.geom,
    )

    target = compute_forward_correction_target(measurement, path, 0, config)
    anchor = path.sample(path.project(measurement.X2, measurement.Y2, 0).station_m)

    assert target.anchor_x_m == pytest.approx(anchor.x)
    assert target.anchor_y_m == pytest.approx(anchor.y)
    assert target.target_heading_rad == pytest.approx(anchor.theta_ref)
    assert np.hypot(target.target_x_m - anchor.x, target.target_y_m - anchor.y) == pytest.approx(50.0)


def test_forward_correction_delays_reverse_path_until_transition():
    config = TrailerLtvMpcConfig()
    controller = object()
    supervisor = ForwardCorrectionSupervisor(controller, config)
    measurement = measurement_from_repo_state([0.0, 0.0, 0.0, 0.0], config.geom)
    path = make_validation_path("straight", "reverse")

    supervisor.start(measurement, path, 0)

    assert supervisor.state.forward_path is not None
    assert supervisor.state.reverse_path is None


def test_reverse_anchor_phase_resets_virtual_input_history():
    config = TrailerLtvMpcConfig()
    measurement = measurement_from_repo_state([0.0, 0.0, 0.0, 0.0], config.geom)
    target = CorrectionTarget(15.0, 0.0, 0.0, 15.0, 0.0)
    controller = _RecordingController(config)
    supervisor = ForwardCorrectionSupervisor(controller, config)
    supervisor.state = ForwardCorrectionState(
        active=True,
        phase="reverse_to_anchor",
        target=target,
        reverse_path=build_locked_anchor_reverse_path(target, measurement),
        reverse_u_prev=np.array([0.0, -config.V2_min_abs_mps]),
    )

    stale_u_prev = np.array([np.deg2rad(-45.0), -0.5])
    output = supervisor.step(measurement.explicit_state, stale_u_prev, make_validation_path("straight", "reverse"), 0)

    np.testing.assert_allclose(controller.u_prev_seen, [0.0, -config.V2_min_abs_mps])
    assert output.debug["mode"] == "forward_correction"
    assert output.debug["phase"] == "reverse_to_anchor"


def test_reverse_anchor_path_offsets_current_pose_past_start_ramp():
    config = TrailerLtvMpcConfig(V2_profile_start_ramp_m=6.0)
    measurement = measurement_from_repo_state([0.0, 0.0, 0.0, 0.0], config.geom)
    target = CorrectionTarget(15.0, 0.0, 0.0, 15.0, 0.0)

    path = build_locked_anchor_reverse_path(target, measurement, config)
    projection = path.project(measurement.X2, measurement.Y2, 0)

    assert projection.station_m == pytest.approx(config.V2_profile_start_ramp_m, abs=0.2)


class _RecordingController:
    def __init__(self, config):
        self.config = config
        self.u_prev_seen = None

    def step(self, plant_state_explicit, u_prev, V2_reference, path_reference, search_start_idx=0):
        self.u_prev_seen = np.asarray(u_prev, dtype=float)
        command = ControllerCommand(delta_f=0.0, V1=0.0, delta_T=0.0, V2=-self.config.V2_min_abs_mps)
        return ControllerOutput(command, {"ref": None}, search_start_idx)
