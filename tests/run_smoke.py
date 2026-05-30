import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from examples.demo_planner import straight_path
from trailer_ltv_mpc import TrailerLtvMpcConfig, TrailerLtvMpcController
from trailer_ltv_mpc.forward_correction import (
    CorrectionTarget,
    build_locked_forward_path,
    pure_pursuit_forward_step,
)
from trailer_ltv_mpc.geometry import measurement_from_repo_state
from trailer_ltv_mpc.ltv_model import build_trailer_ltv_mpc_model
from trailer_ltv_mpc.mapping import compute_admissible_delta_T_bounds, map_virtual_to_actual
from trailer_ltv_mpc.qp_solver import build_qp_problem
from trailer_ltv_mpc.reference_builder import generate_trailer_ltv_mpc_reference
from trailer_ltv_mpc.speed_profile import apply_start_end_speed_profile


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    config = TrailerLtvMpcConfig(N=8)
    measurement = measurement_from_repo_state([0.0, 0.0, 0.0, 0.0], config.geom)
    check(np.isclose(measurement.X1, config.geom.L2 - config.geom.L1c), "geometry reconstruction")

    mapping = map_virtual_to_actual(0.0, 0.0, -1.0, config.geom, config.mapping_denominator_min)
    check(np.isclose(mapping.delta_f, 0.0), "mapping delta_f")
    check(np.isclose(mapping.V1, -1.0), "mapping V1")

    delta_min, delta_max, _ = compute_admissible_delta_T_bounds(0.0, config, -1.0)
    check(delta_min < 0.0 < delta_max, "delta_T bounds include zero")

    path = straight_path(length_m=20.0, direction="reverse")
    profile_start = apply_start_end_speed_profile(-1.0, path, config, 0.0)
    profile_mid = apply_start_end_speed_profile(-1.0, path, config, 10.0)
    check(np.isclose(profile_start.reference_profile, 0.0), "start profile")
    check(np.isclose(profile_mid.reference_profile, -1.0), "middle profile")

    X0 = np.array([0.0, 0.0, np.pi])
    u_prev = np.array([0.0, -0.2])
    ref = generate_trailer_ltv_mpc_reference(X0, u_prev, -1.0, path, config, 0)
    model = build_trailer_ltv_mpc_model(ref, config)
    check(model.Ad.shape == (3, 3, config.N), "model Ad shape")

    umin = np.vstack([-0.5 * np.ones(config.N), -1.5 * np.ones(config.N)])
    umax = np.vstack([0.5 * np.ones(config.N), np.zeros(config.N)])
    problem = build_qp_problem(
        model,
        config.Q_rev,
        config.Qf_rev,
        config.R,
        u_prev,
        X0,
        umin,
        umax,
        np.array([-0.1, -0.1]),
        np.array([0.1, 0.1]),
        config.Rd,
    )
    check(problem.H.shape == (2 * config.N, 2 * config.N), "QP H shape")

    fc_target = CorrectionTarget(15.0, 0.0, 0.0, 0.0, 0.0)
    fc_path = build_locked_forward_path(fc_target, measurement, config)
    pp_output = pure_pursuit_forward_step(measurement, fc_path, 0, config)
    check(np.isfinite(pp_output.command.delta_f), "pure pursuit delta_f finite")

    if importlib.util.find_spec("osqp") is not None:
        controller = TrailerLtvMpcController(config)
        plant_state = np.array([-9.739, 0.0, np.pi, 0.0, 0.0, np.pi])
        output = controller.step(plant_state, u_prev, -1.0, path, 0)
        check(np.isfinite(output.command.delta_f), "controller delta_f finite")
        check(np.isfinite(output.command.V2), "controller V2 finite")
    else:
        print("OSQP not installed; skipped closed-loop controller solve smoke.")

    print("trailer_ltv_mpc smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
