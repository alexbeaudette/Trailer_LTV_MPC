import numpy as np

from trailer_controller.config import TrailerLtvMpcConfig
from trailer_controller.forward_correction import build_locked_forward_path, pure_pursuit_forward_step
from trailer_controller.forward_correction import CorrectionTarget
from trailer_controller.geometry import measurement_from_repo_state


def test_pure_pursuit_forward_correction_outputs_finite_command():
    config = TrailerLtvMpcConfig()
    measurement = measurement_from_repo_state([0.0, 0.0, 0.0, 0.0], config.geom)
    target = CorrectionTarget(15.0, 0.0, 0.0, 0.0, 0.0)
    path = build_locked_forward_path(target, measurement, config)
    output = pure_pursuit_forward_step(measurement, path, 0, config)
    assert np.isfinite(output.command.delta_f)
    assert np.isfinite(output.command.V1)
    assert abs(output.command.delta_f) <= config.delta_f_max_rad
