import numpy as np

from trailer_controller.config import TrailerLtvMpcConfig
from trailer_controller.reference_builder import generate_trailer_ltv_mpc_reference

from examples.demo_planner import straight_path


def test_reference_builder_shapes():
    config = TrailerLtvMpcConfig(N=8)
    path = straight_path(length_m=20.0, direction="reverse")
    ref = generate_trailer_ltv_mpc_reference(
        np.array([0.0, 0.0, np.pi]), np.array([0.0, -0.2]), -1.0, path, config, 0
    )
    assert ref.X_ref.shape == (3, 8)
    assert ref.U_ref.shape == (2, 8)
    assert ref.V2_ref.shape == (8,)
