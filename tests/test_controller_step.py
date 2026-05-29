import importlib.util

import numpy as np
import pytest

from trailer_controller import TrailerLtvMpcConfig, TrailerLtvMpcController

from examples.demo_planner import straight_path


@pytest.mark.skipif(importlib.util.find_spec("osqp") is None, reason="OSQP not installed")
def test_controller_step_returns_finite_command():
    config = TrailerLtvMpcConfig(N=8)
    controller = TrailerLtvMpcController(config)
    path = straight_path(length_m=20.0, direction="reverse")
    plant_state = np.array([9.739, 0.0, 0.0, 0.0, 0.0, np.pi])
    output = controller.step(plant_state, np.array([0.0, -0.2]), -1.0, path, 0)
    assert np.isfinite(output.command.delta_f)
    assert np.isfinite(output.command.V1)
    assert np.isfinite(output.command.delta_T)
    assert np.isfinite(output.command.V2)
