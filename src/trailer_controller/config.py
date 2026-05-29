from dataclasses import dataclass, field

import numpy as np

from .geometry import Geometry


def _diag(values):
    return np.diag(np.asarray(values, dtype=float))


@dataclass
class TrailerLtvMpcConfig:
    geom: Geometry = field(default_factory=Geometry)
    Ts: float = 0.05
    N: int = 40
    nx: int = 3
    nu: int = 2

    V2_reference_mps: float = -1.0
    V2_fixed_mps: float = -1.0
    V2_min_abs_mps: float = 0.2
    V2_max_abs_mps: float = 1.5
    V2_rate_max_mps2: float = 0.5

    enable_adaptive_V2_reference: bool = True
    adaptive_V2_min_scale: float = 0.35
    adaptive_V2_cte_start_m: float = 0.20
    adaptive_V2_cte_full_m: float = 1.00
    adaptive_V2_heading_start_rad: float = np.deg2rad(5.0)
    adaptive_V2_heading_full_rad: float = np.deg2rad(25.0)
    adaptive_V2_gamma_start_rad: float = np.deg2rad(20.0)
    adaptive_V2_gamma_full_rad: float = np.deg2rad(40.0)

    enable_start_end_V2_profile: bool = True
    V2_profile_start_ramp_m: float = 6.0
    V2_profile_end_ramp_m: float = 8.0
    V2_profile_stop_tolerance_mps: float = 0.05

    enable_forward_correction: bool = False
    forward_correction_terminal_heading_threshold_rad: float = np.deg2rad(15.0)
    forward_correction_gamma_trigger_rad: float = np.deg2rad(50.0)
    forward_correction_gamma_exit_rad: float = np.deg2rad(35.0)
    forward_correction_station_lookahead_m: float = 1.0
    forward_correction_target_distance_m: float = 50.0
    forward_correction_target_reached_tolerance_m: float = 1.0
    forward_correction_target_pass_radius_m: float = np.nan
    forward_correction_anchor_reached_tolerance_m: float = 1.0
    forward_correction_method: str = "pure_pursuit"
    forward_correction_tracking_point: str = "trailer_rear_axle"
    forward_correction_path_start_offset_m: float = np.nan
    forward_correction_path_start_offset_profile_fraction: float = 0.5
    forward_correction_forward_path_lookahead_m: float = 3.0
    forward_correction_V1_mps: float = 1.0

    delta_T_min_rad: float = -np.deg2rad(80.0)
    delta_T_max_rad: float = np.deg2rad(80.0)
    delta_T_rate_max_radps: float = np.deg2rad(90.0)
    delta_f_max_rad: float = np.pi / 6.0
    mapping_denominator_min: float = 1.0e-3
    delta_T_bound_grid_size: int = 4001

    Q: np.ndarray = field(default_factory=lambda: _diag([100.0, 100.0, 100.0]))
    Qf: np.ndarray = field(default_factory=lambda: _diag([1000.0, 1000.0, 1000.0]))
    Q_rev: np.ndarray = field(default_factory=lambda: _diag([100.0, 100.0, 100.0]))
    Qf_rev: np.ndarray = field(default_factory=lambda: _diag([1000.0, 1000.0, 1000.0]))
    Q_fwd: np.ndarray = field(default_factory=lambda: _diag([150.0, 150.0, 10.0]))
    Qf_fwd: np.ndarray = field(default_factory=lambda: _diag([200.0, 200.0, 10.0]))
    R: np.ndarray = field(default_factory=lambda: _diag([15.0, 80.0]))
    Rd: np.ndarray = field(default_factory=lambda: _diag([10.0, 50.0]))

    def weights_for_motion(self, motion_sign: float):
        if motion_sign > 0.0:
            return self.Q_fwd, self.Qf_fwd, "forward"
        return self.Q_rev, self.Qf_rev, "reverse"
