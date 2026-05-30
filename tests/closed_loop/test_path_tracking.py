"""Closed-loop path tracking validation for forward and reverse motion."""

import importlib.util

import pytest

from trailer_ltv_mpc import TrailerLtvMpcConfig

from validation.closed_loop import assert_closed_loop_result, run_closed_loop_path
from validation.path_generation import make_validation_path


@pytest.mark.skipif(importlib.util.find_spec("osqp") is None, reason="OSQP not installed")
@pytest.mark.parametrize("path_kind", ["straight", "arc", "spline"])
@pytest.mark.parametrize("direction, direction_sign", [("forward", 1.0), ("reverse", -1.0)])
def test_controller_tracks_path_family(path_kind, direction, direction_sign):
    config = TrailerLtvMpcConfig(N=8)
    path = make_validation_path(path_kind, direction)

    result = run_closed_loop_path(
        path,
        direction_sign,
        config,
        metadata={"path_kind": path_kind, "direction": direction, "steps": 120, "ds": 0.2},
    )

    assert_closed_loop_result(result, direction_sign, config)
