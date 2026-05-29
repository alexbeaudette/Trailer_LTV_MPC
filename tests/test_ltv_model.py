import numpy as np

from trailer_controller.config import TrailerLtvMpcConfig
from trailer_controller.ltv_model import build_trailer_ltv_mpc_model
from trailer_controller.reference_builder import generate_trailer_ltv_mpc_reference

from examples.demo_planner import straight_path


def test_ltv_model_shapes():
    config = TrailerLtvMpcConfig(N=6)
    path = straight_path(length_m=20.0, direction="reverse")
    ref = generate_trailer_ltv_mpc_reference(
        np.array([0.0, 0.0, np.pi]), np.array([0.0, -0.2]), -1.0, path, config, 0
    )
    model = build_trailer_ltv_mpc_model(ref, config)
    assert model.Ad.shape == (3, 3, 6)
    assert model.Bd.shape == (3, 2, 6)
    assert model.gd.shape == (3, 6)
