from dataclasses import dataclass

import numpy as np

from .math_utils import wrap_to_pi


@dataclass(frozen=True)
class Geometry:
    L1: float = 3.261
    L1c: float = 0.261
    L2: float = 10.0

    @property
    def tractor_wheelbase_m(self) -> float:
        return self.L1

    @property
    def hitch_offset_m(self) -> float:
        return self.L1c

    @property
    def trailer_wheelbase_m(self) -> float:
        return self.L2


@dataclass(frozen=True)
class Measurement:
    X1: float
    Y1: float
    psi1: float
    X2: float
    Y2: float
    psi2: float
    Xh: float
    Yh: float
    gamma: float

    @property
    def explicit_state(self) -> np.ndarray:
        return np.array([self.X1, self.Y1, self.psi1, self.X2, self.Y2, self.psi2], dtype=float)

    @property
    def trailer_state(self) -> np.ndarray:
        return np.array([self.X2, self.Y2, self.psi2], dtype=float)


def measurement_from_explicit_state(plant_state_explicit, geom: Geometry) -> Measurement:
    state = np.asarray(plant_state_explicit, dtype=float).reshape(6)
    X1, Y1, psi1, X2, Y2, psi2 = state
    Xh = X2 + geom.L2 * np.cos(psi2)
    Yh = Y2 + geom.L2 * np.sin(psi2)
    gamma = float(wrap_to_pi(psi1 - psi2))
    return Measurement(X1, Y1, psi1, X2, Y2, psi2, Xh, Yh, gamma)


def measurement_from_repo_state(repo_state, geom: Geometry) -> Measurement:
    state = np.asarray(repo_state, dtype=float).reshape(4)
    X2, Y2, psi1, psi2 = state
    Xh = X2 + geom.L2 * np.cos(psi2)
    Yh = Y2 + geom.L2 * np.sin(psi2)
    X1 = Xh - geom.L1c * np.cos(psi1)
    Y1 = Yh - geom.L1c * np.sin(psi1)
    gamma = float(wrap_to_pi(psi1 - psi2))
    return Measurement(X1, Y1, psi1, X2, Y2, psi2, Xh, Yh, gamma)


def explicit_state_to_trailer_state(plant_state_explicit) -> np.ndarray:
    state = np.asarray(plant_state_explicit, dtype=float).reshape(6)
    return np.array([state[3], state[4], state[5]], dtype=float)
