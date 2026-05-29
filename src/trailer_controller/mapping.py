from dataclasses import dataclass

import numpy as np

from .config import TrailerLtvMpcConfig
from .geometry import Geometry
from .math_utils import wrap_to_pi


@dataclass(frozen=True)
class MappingTerms:
    denominator: np.ndarray
    numerator: np.ndarray
    convention: str


@dataclass(frozen=True)
class MappingResult:
    delta_f: float
    V1: float
    gamma: float
    delta_T: float
    V2: float
    motion_sign: float
    convention: str
    numerator: float
    denominator: float


def compute_virtual_to_actual_mapping_terms(gamma, delta_T, motion_sign: float) -> MappingTerms:
    gamma = wrap_to_pi(gamma)
    tan_delta_T = np.tan(delta_T)
    if motion_sign >= 0.0:
        denominator = np.cos(gamma) + np.sin(gamma) * tan_delta_T
        numerator = -np.sin(gamma) + np.cos(gamma) * tan_delta_T
        convention = "forward"
    else:
        denominator = np.cos(gamma) - np.sin(gamma) * tan_delta_T
        numerator = np.sin(gamma) - np.cos(gamma) * tan_delta_T
        convention = "reverse_stabilizing"
    return MappingTerms(np.asarray(denominator), np.asarray(numerator), convention)


def map_virtual_to_actual(
    delta_T: float,
    gamma: float,
    V2: float,
    geom: Geometry,
    mapping_denominator_min: float,
) -> MappingResult:
    motion_sign = float(np.sign(V2) or 1.0)
    terms = compute_virtual_to_actual_mapping_terms(gamma, delta_T, motion_sign)
    denominator = float(terms.denominator)
    numerator = float(terms.numerator)
    if abs(denominator) <= mapping_denominator_min:
        raise ValueError("Mapping denominator is too small.")
    delta_f = float(np.arctan2(geom.L1 * numerator, geom.L1c * denominator))
    V1 = float(V2 * denominator)
    return MappingResult(
        delta_f=delta_f,
        V1=V1,
        gamma=float(wrap_to_pi(gamma)),
        delta_T=float(delta_T),
        V2=float(V2),
        motion_sign=motion_sign,
        convention=terms.convention,
        numerator=numerator,
        denominator=denominator,
    )


def compute_admissible_delta_T_bounds(gamma: float, config: TrailerLtvMpcConfig, motion_sign: float):
    motion_sign = float(np.sign(motion_sign) or 1.0)
    gamma = float(wrap_to_pi(gamma))
    grid_size = int(round(config.delta_T_bound_grid_size))
    if grid_size < 3:
        raise ValueError("delta_T_bound_grid_size must be at least 3.")

    guard_min = min(config.delta_T_min_rad, config.delta_T_max_rad)
    guard_max = max(config.delta_T_min_rad, config.delta_T_max_rad)
    delta_T_grid = np.linspace(guard_min, guard_max, grid_size)
    terms = compute_virtual_to_actual_mapping_terms(gamma, delta_T_grid, motion_sign)
    delta_f_grid = np.arctan2(config.geom.L1 * terms.numerator, config.geom.L1c * terms.denominator)
    feasible = (
        np.isfinite(delta_f_grid)
        & np.isfinite(terms.denominator)
        & (np.abs(terms.denominator) > config.mapping_denominator_min)
        & (np.abs(delta_f_grid) <= config.delta_f_max_rad + 1.0e-10)
    )
    if not np.any(feasible):
        raise ValueError("No admissible delta_T interval exists.")
    feasible_idx = np.flatnonzero(feasible)
    first_idx = int(feasible_idx[0])
    last_idx = int(feasible_idx[-1])
    if np.any(~feasible[first_idx : last_idx + 1]):
        raise ValueError("Admissible delta_T set is disconnected.")
    if first_idx == last_idx:
        raise ValueError("Admissible delta_T interval collapsed to a point.")
    info = {
        "gamma": gamma,
        "motion_sign": motion_sign,
        "mapping_convention": terms.convention,
        "delta_T_guard_min_rad": guard_min,
        "delta_T_guard_max_rad": guard_max,
        "delta_f_limit_rad": config.delta_f_max_rad,
        "mapping_denominator_min": config.mapping_denominator_min,
        "grid_resolution_rad": float(delta_T_grid[1] - delta_T_grid[0]),
        "num_feasible_samples": int(np.count_nonzero(feasible)),
        "delta_f_feasible_min_rad": float(np.min(delta_f_grid[feasible])),
        "delta_f_feasible_max_rad": float(np.max(delta_f_grid[feasible])),
    }
    return float(delta_T_grid[first_idx]), float(delta_T_grid[last_idx]), info
