import numpy as np
import pytest

from trailer_ltv_mpc.config import TrailerLtvMpcConfig
from trailer_ltv_mpc.mapping import compute_admissible_delta_T_bounds, map_virtual_to_actual
from trailer_ltv_mpc.models import recover_virtual_input_from_full_plant


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


def test_virtual_to_actual_recovers_nonzero_forward_virtual_input():
    config = TrailerLtvMpcConfig()
    repo_state = np.array([0.0, 0.0, 0.25, -0.10])
    delta_T_cmd = 0.18
    V2_cmd = 1.1

    mapped = map_virtual_to_actual(delta_T_cmd, 0.35, V2_cmd, config.geom, config.mapping_denominator_min)
    recovered = recover_virtual_input_from_full_plant(repo_state, mapped.delta_f, mapped.V1, config.geom)

    assert recovered.delta_T == pytest.approx(delta_T_cmd)
    assert recovered.V2 == pytest.approx(V2_cmd)


def test_reverse_stabilizing_mapping_uses_documented_convention():
    config = TrailerLtvMpcConfig()
    gamma = 0.35
    delta_T_cmd = -0.22
    V2_cmd = -0.9

    mapped = map_virtual_to_actual(delta_T_cmd, gamma, V2_cmd, config.geom, config.mapping_denominator_min)

    expected_denominator = np.cos(gamma) - np.sin(gamma) * np.tan(delta_T_cmd)
    expected_numerator = np.sin(gamma) - np.cos(gamma) * np.tan(delta_T_cmd)
    assert mapped.denominator == pytest.approx(expected_denominator)
    assert mapped.numerator == pytest.approx(expected_numerator)
    assert mapped.convention == "reverse_stabilizing"


def test_reverse_stabilizing_mapping_recovers_commanded_speed():
    config = TrailerLtvMpcConfig()
    repo_state = np.array([0.0, 0.0, 0.25, -0.10])
    delta_T_cmd = -0.22
    V2_cmd = -0.9

    mapped = map_virtual_to_actual(delta_T_cmd, 0.35, V2_cmd, config.geom, config.mapping_denominator_min)
    recovered = recover_virtual_input_from_full_plant(repo_state, mapped.delta_f, mapped.V1, config.geom)

    assert recovered.V2 == pytest.approx(V2_cmd)


def test_mapping_with_steering_limit_preserves_speed():
    config = TrailerLtvMpcConfig()
    repo_state = np.array([0.0, 0.0, 0.25, -0.10])

    mapped = map_virtual_to_actual(
        delta_T=0.8,
        gamma=0.35,
        V2=-0.9,
        geom=config.geom,
        mapping_denominator_min=config.mapping_denominator_min,
        delta_f_limit_rad=config.delta_f_max_rad,
    )
    recovered = recover_virtual_input_from_full_plant(repo_state, mapped.delta_f, mapped.V1, config.geom)

    assert abs(mapped.delta_f) <= config.delta_f_max_rad
    assert recovered.V2 == pytest.approx(-0.9)
