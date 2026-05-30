"""Virtual trailer steering to physical truck command mapping."""

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
    delta_f_limit_rad: float | None = None,
) -> MappingResult:
    motion_sign = float(np.sign(V2) or 1.0)
    terms = compute_virtual_to_actual_mapping_terms(gamma, delta_T, motion_sign)
    denominator = float(terms.denominator)
    numerator = float(terms.numerator)
    if abs(denominator) <= mapping_denominator_min:
        raise ValueError("Mapping denominator is too small.")
    delta_f = float(np.arctan2(geom.L1 * numerator, geom.L1c * denominator))
    if delta_f_limit_rad is not None:
        delta_f = float(np.clip(delta_f, -abs(delta_f_limit_rad), abs(delta_f_limit_rad)))
    speed_gain = _plant_trailer_speed_gain(gamma, delta_f, geom)
    if abs(speed_gain) <= mapping_denominator_min:
        raise ValueError("Mapping speed gain is too small.")
    V1 = float(V2 / speed_gain)
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


def _plant_trailer_speed_gain(gamma: float, delta_f: float, geom: Geometry) -> float:
    gamma = float(wrap_to_pi(gamma))
    return float(np.cos(gamma) - (geom.L1c / geom.L1) * np.sin(gamma) * np.tan(delta_f))


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
    delta_T_min = _refine_feasible_boundary(
        delta_T_grid,
        feasible,
        first_idx,
        gamma,
        motion_sign,
        config,
        side="left",
    )
    delta_T_max = _refine_feasible_boundary(
        delta_T_grid,
        feasible,
        last_idx,
        gamma,
        motion_sign,
        config,
        side="right",
    )
    if delta_T_min >= delta_T_max:
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
    return float(delta_T_min), float(delta_T_max), info


def _refine_feasible_boundary(delta_T_grid, feasible, feasible_idx, gamma, motion_sign, config, side: str) -> float:
    if side == "left":
        if feasible_idx == 0:
            return float(delta_T_grid[feasible_idx])
        infeasible_value = float(delta_T_grid[feasible_idx - 1])
        feasible_value = float(delta_T_grid[feasible_idx])
        return _bisect_boundary(infeasible_value, feasible_value, gamma, motion_sign, config, feasible_on_right=True)
    if feasible_idx == len(delta_T_grid) - 1:
        return float(delta_T_grid[feasible_idx])
    feasible_value = float(delta_T_grid[feasible_idx])
    infeasible_value = float(delta_T_grid[feasible_idx + 1])
    return _bisect_boundary(feasible_value, infeasible_value, gamma, motion_sign, config, feasible_on_right=False)


def _bisect_boundary(left, right, gamma, motion_sign, config, feasible_on_right: bool) -> float:
    left = float(left)
    right = float(right)
    for _ in range(60):
        middle = 0.5 * (left + right)
        if _delta_T_is_feasible(middle, gamma, motion_sign, config) == feasible_on_right:
            right = middle
        else:
            left = middle
    return right if feasible_on_right else left


def _delta_T_is_feasible(delta_T, gamma, motion_sign, config) -> bool:
    terms = compute_virtual_to_actual_mapping_terms(gamma, delta_T, motion_sign)
    denominator = float(terms.denominator)
    numerator = float(terms.numerator)
    if not np.isfinite(denominator) or abs(denominator) <= config.mapping_denominator_min:
        return False
    delta_f = float(np.arctan2(config.geom.L1 * numerator, config.geom.L1c * denominator))
    return bool(np.isfinite(delta_f) and abs(delta_f) <= config.delta_f_max_rad + 1.0e-10)
