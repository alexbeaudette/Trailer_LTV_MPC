import numpy as np

from .geometry import Geometry
from .math_utils import wrap_to_pi


def trailer_model_continuous(x, delta_T: float, V2: float, geom: Geometry) -> np.ndarray:
    X2, Y2, psi2 = np.asarray(x, dtype=float).reshape(3)
    return np.array(
        [
            V2 * np.cos(psi2),
            V2 * np.sin(psi2),
            (V2 / geom.L2) * np.tan(delta_T),
        ],
        dtype=float,
    )


def propagate_trailer_virtual_state_one_step(x, delta_T: float, V2: float, dt: float, geom: Geometry):
    x_next = np.asarray(x, dtype=float).reshape(3) + dt * trailer_model_continuous(x, delta_T, V2, geom)
    x_next[2] = wrap_to_pi(x_next[2])
    return x_next


def tractor_trailer_kinematics(repo_state, delta_f: float, V1: float, geom: Geometry):
    X2, Y2, psi1, psi2 = np.asarray(repo_state, dtype=float).reshape(4)
    gamma = wrap_to_pi(psi1 - psi2)
    tan_delta = np.tan(delta_f)
    V2 = V1 * (np.cos(gamma) - (geom.L1c / geom.L1) * np.sin(gamma) * tan_delta)
    psi1_dot = (V1 / geom.L1) * tan_delta
    psi2_dot = (V1 / geom.L2) * (np.sin(gamma) + (geom.L1c / geom.L1) * np.cos(gamma) * tan_delta)
    return np.array([V2 * np.cos(psi2), V2 * np.sin(psi2), psi1_dot, psi2_dot], dtype=float)


def propagate_full_plant_one_step(repo_state, delta_f: float, V1: float, dt: float, geom: Geometry):
    x_next = np.asarray(repo_state, dtype=float).reshape(4) + dt * tractor_trailer_kinematics(
        repo_state, delta_f, V1, geom
    )
    x_next[2] = wrap_to_pi(x_next[2])
    x_next[3] = wrap_to_pi(x_next[3])
    return x_next
