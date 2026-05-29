import numpy as np

from trailer_controller.geometry import Geometry, measurement_from_repo_state


def test_measurement_reconstruction_front_hitch_geometry():
    geom = Geometry()
    measurement = measurement_from_repo_state([0.0, 0.0, 0.0, 0.0], geom)
    assert np.isclose(measurement.Xh, geom.L2)
    assert np.isclose(measurement.Yh, 0.0)
    assert np.isclose(measurement.X1, geom.L2 - geom.L1c)
    assert np.isclose(measurement.gamma, 0.0)
