from dataclasses import dataclass

import numpy as np

from .config import TrailerLtvMpcConfig
from .geometry import explicit_state_to_trailer_state, measurement_from_explicit_state
from .ltv_model import build_trailer_ltv_mpc_model
from .mapping import compute_admissible_delta_T_bounds, map_virtual_to_actual
from .path_reference import PathReference
from .qp_solver import QpSolver, build_qp_problem
from .reference_builder import generate_trailer_ltv_mpc_reference
from .speed_profile import compute_effective_V2_reference


@dataclass(frozen=True)
class ControllerCommand:
    delta_f: float
    V1: float
    delta_T: float
    V2: float


@dataclass(frozen=True)
class ControllerOutput:
    command: ControllerCommand
    debug: dict
    search_start_idx: int


class TrailerLtvMpcController:
    def __init__(self, config: TrailerLtvMpcConfig | None = None, solver: QpSolver | None = None):
        self.config = config or TrailerLtvMpcConfig()
        self.solver = solver or QpSolver()

    def step(
        self,
        plant_state_explicit,
        u_prev,
        V2_reference: float,
        path_reference: PathReference,
        search_start_idx: int = 0,
    ) -> ControllerOutput:
        config = self.config
        measurement = measurement_from_explicit_state(plant_state_explicit, config.geom)
        X_mpc = explicit_state_to_trailer_state(plant_state_explicit)
        u_prev = np.asarray(u_prev, dtype=float).reshape(2)

        V2_effective, adaptive = compute_effective_V2_reference(
            X_mpc, measurement, V2_reference, path_reference, config, search_start_idx
        )
        motion_sign = float(np.sign(V2_effective) or np.sign(V2_reference) or 1.0)
        delta_T_min, delta_T_max, constraint_info = compute_admissible_delta_T_bounds(
            measurement.gamma, config, motion_sign
        )
        ref = generate_trailer_ltv_mpc_reference(
            X_mpc, u_prev, V2_effective, path_reference, config, search_start_idx
        )
        model = build_trailer_ltv_mpc_model(ref, config)
        V2_min_profile, V2_max_profile = compute_stagewise_directional_V2_bounds(
            ref.V2_ref, V2_effective, ref.V2_profile, config, path_reference, u_prev[1]
        )
        umin = np.vstack([delta_T_min * np.ones(config.N), V2_min_profile])
        umax = np.vstack([delta_T_max * np.ones(config.N), V2_max_profile])
        delta_umin = np.array(
            [-config.Ts * config.delta_T_rate_max_radps, -config.Ts * config.V2_rate_max_mps2]
        )
        delta_umax = np.array(
            [config.Ts * config.delta_T_rate_max_radps, config.Ts * config.V2_rate_max_mps2]
        )
        Q, Qf, weight_mode = config.weights_for_motion(motion_sign)
        problem = build_qp_problem(
            model, Q, Qf, config.R, u_prev, X_mpc, umin, umax, delta_umin, delta_umax, config.Rd
        )
        solution = self.solver.solve(problem)
        u_plan = solution.U.reshape((config.nu, config.N), order="F")
        delta_T_cmd = float(u_plan[0, 0])
        V2_cmd = float(u_plan[1, 0])
        mapping = map_virtual_to_actual(
            delta_T_cmd,
            measurement.gamma,
            V2_cmd,
            config.geom,
            config.mapping_denominator_min,
            config.delta_f_max_rad,
        )
        command = ControllerCommand(mapping.delta_f, mapping.V1, delta_T_cmd, V2_cmd)
        debug = {
            "measurement": measurement,
            "X_mpc": X_mpc,
            "ref": ref,
            "model": model,
            "u_plan": u_plan,
            "adaptive_V2": adaptive,
            "V2_reference_effective": V2_effective,
            "V2_min_profile": V2_min_profile,
            "V2_max_profile": V2_max_profile,
            "constraint_info": constraint_info,
            "weight_mode": weight_mode,
            "mapping": mapping,
            "solver": solution,
        }
        return ControllerOutput(command, debug, int(ref.i0))


def compute_stagewise_directional_V2_bounds(
    V2_reference_profile,
    V2_direction_reference: float,
    profile,
    config: TrailerLtvMpcConfig,
    path_reference: PathReference,
    V2_prev: float,
):
    min_abs = abs(config.V2_min_abs_mps)
    max_abs = abs(config.V2_max_abs_mps)
    if min_abs > max_abs:
        raise ValueError("V2_min_abs_mps must be <= V2_max_abs_mps.")
    direction_sign = float(np.sign(V2_direction_reference))
    if direction_sign == 0.0:
        finite_ref = np.asarray(V2_reference_profile)[np.asarray(V2_reference_profile) != 0.0]
        direction_sign = float(np.sign(finite_ref[0]) if finite_ref.size else 1.0)
    active = np.asarray(profile.active, dtype=bool)
    N = len(V2_reference_profile)
    V2_min = np.zeros(N)
    V2_max = np.zeros(N)
    for stage_idx in range(N):
        allow_zero = bool(active[stage_idx])
        if direction_sign < 0.0:
            V2_min[stage_idx] = -max_abs
            V2_max[stage_idx] = 0.0 if allow_zero else -min_abs
        else:
            V2_min[stage_idx] = 0.0 if allow_zero else min_abs
            V2_max[stage_idx] = max_abs
        if path_reference.type == "forward_correction_forward_to_target":
            reachable = abs(V2_prev) + (stage_idx + 1) * config.Ts * config.V2_rate_max_mps2
            if direction_sign < 0.0 and V2_max[stage_idx] < 0.0:
                V2_max[stage_idx] = max(V2_max[stage_idx], -reachable)
            elif direction_sign >= 0.0 and V2_min[stage_idx] > 0.0:
                V2_min[stage_idx] = min(V2_min[stage_idx], reachable)
    return V2_min, V2_max
