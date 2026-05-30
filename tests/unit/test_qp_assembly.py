import numpy as np

from trailer_ltv_mpc.config import TrailerLtvMpcConfig
from trailer_ltv_mpc.ltv_model import build_trailer_ltv_mpc_model
from trailer_ltv_mpc.qp_solver import build_qp_problem
from trailer_ltv_mpc.reference_builder import generate_trailer_ltv_mpc_reference

from examples.demo_planner import straight_path


def test_qp_problem_dimensions():
    config = TrailerLtvMpcConfig(N=5)
    path = straight_path(length_m=20.0, direction="reverse")
    X0 = np.array([0.0, 0.0, np.pi])
    u_prev = np.array([0.0, -0.2])
    ref = generate_trailer_ltv_mpc_reference(X0, u_prev, -1.0, path, config, 0)
    model = build_trailer_ltv_mpc_model(ref, config)
    umin = np.vstack([-0.5 * np.ones(config.N), -1.5 * np.ones(config.N)])
    umax = np.vstack([0.5 * np.ones(config.N), -0.05 * np.ones(config.N)])
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
    assert problem.H.shape == (10, 10)
    assert problem.A.shape[1] == 10
    assert problem.lower.shape == problem.upper.shape
