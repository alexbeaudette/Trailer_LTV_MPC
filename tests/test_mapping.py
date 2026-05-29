import numpy as np

from trailer_controller.config import TrailerLtvMpcConfig
from trailer_controller.mapping import compute_admissible_delta_T_bounds, map_virtual_to_actual


def test_virtual_to_actual_zero_steering_forward():
    config = TrailerLtvMpcConfig()
    result = map_virtual_to_actual(0.0, 0.0, 1.0, config.geom, config.mapping_denominator_min)
    assert np.isclose(result.delta_f, 0.0)
    assert np.isclose(result.V1, 1.0)
    assert result.convention == "forward"


def test_virtual_to_actual_zero_steering_reverse():
    config = TrailerLtvMpcConfig()
    result = map_virtual_to_actual(0.0, 0.0, -1.0, config.geom, config.mapping_denominator_min)
    assert np.isclose(result.delta_f, 0.0)
    assert np.isclose(result.V1, -1.0)
    assert result.convention == "reverse_stabilizing"


def test_admissible_delta_T_bounds_include_zero():
    config = TrailerLtvMpcConfig()
    delta_min, delta_max, info = compute_admissible_delta_T_bounds(0.0, config, -1.0)
    assert delta_min < 0.0 < delta_max
    assert info["num_feasible_samples"] > 0
