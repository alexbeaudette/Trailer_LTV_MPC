"""Path convention checks for closed-loop validation fixtures."""

import numpy as np
import pytest

from validation.path_generation import make_validation_path


@pytest.mark.parametrize("path_kind", ["straight", "arc", "sinusoid", "spline", "harsh_turn"])
@pytest.mark.parametrize("direction, direction_sign, tangent_dot", [("forward", 1.0, 1.0), ("reverse", -1.0, -1.0)])
def test_path_direction_and_heading_convention(path_kind, direction, direction_sign, tangent_dot):
    path = make_validation_path(path_kind, direction)
    dx = np.diff(path.x_r)
    dy = np.diff(path.y_r)
    segment_length = np.hypot(dx, dy)
    tangent_unit = np.column_stack([dx / segment_length, dy / segment_length])
    heading_unit = np.column_stack([np.cos(path.theta_r[:-1]), np.sin(path.theta_r[:-1])])
    dots = np.sum(tangent_unit * heading_unit, axis=1)

    assert np.all(path.dir_r == direction_sign)
    assert np.all(dots * tangent_dot > 0.999)


@pytest.mark.parametrize("path_kind", ["arc", "sinusoid", "spline", "harsh_turn"])
def test_reverse_path_flips_virtual_steering_reference(path_kind):
    forward = make_validation_path(path_kind, "forward")
    reverse = make_validation_path(path_kind, "reverse")

    assert np.allclose(
        forward.delta_T_profile(10.0),
        -reverse.delta_T_profile(10.0),
        atol=1.0e-12,
    )


def test_spline_uses_shared_waypoint_shape():
    path = make_validation_path("spline", "forward")

    assert np.isclose(path.x_r[0], 0.0)
    assert np.isclose(path.y_r[0], 0.0)
    assert np.isclose(path.x_r[-1], 114.0)
    assert np.isclose(path.y_r[-1], 0.0)
    assert np.max(path.y_r) > 8.0
    assert np.min(path.y_r) < -3.0
    assert np.all(np.diff(path.s_r) > 0.0)


def test_sinusoid_has_expected_wave_shape():
    path = make_validation_path("sinusoid", "forward")

    assert path.x_r[0] == pytest.approx(0.0)
    assert path.y_r[0] == pytest.approx(0.0)
    assert path.x_r[-1] == pytest.approx(120.0)
    assert path.y_r[-1] == pytest.approx(6.0 * np.sin(2.0 * np.pi * 120.0 / 45.0))
    assert np.allclose(np.diff(path.x_r), 0.2)
    assert np.max(path.y_r) > 5.9
    assert np.min(path.y_r) < -5.9
    assert np.all(np.diff(path.s_r) > 0.0)


def test_harsh_turn_matches_expected_corner_scale():
    path = make_validation_path("harsh_turn", "reverse")

    assert path.x_r[0] == pytest.approx(0.0)
    assert path.y_r[0] == pytest.approx(0.0)
    assert path.x_r[-1] > 29.0
    assert path.y_r[-1] > 28.0
    assert np.all(path.dir_r == -1.0)
    assert np.all(np.diff(path.s_r) > 0.0)
