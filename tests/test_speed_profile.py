import numpy as np

from trailer_controller.config import TrailerLtvMpcConfig
from trailer_controller.speed_profile import apply_start_end_speed_profile

from examples.demo_planner import straight_path


def test_start_end_profile_scales_start_middle_end():
    config = TrailerLtvMpcConfig()
    path = straight_path(length_m=20.0, direction="forward")
    at_start = apply_start_end_speed_profile(1.0, path, config, 0.0)
    at_middle = apply_start_end_speed_profile(1.0, path, config, 10.0)
    at_end = apply_start_end_speed_profile(1.0, path, config, 20.0)
    assert np.isclose(at_start.reference_profile, 0.0)
    assert np.isclose(at_middle.reference_profile, 1.0)
    assert np.isclose(at_end.reference_profile, 0.0)
