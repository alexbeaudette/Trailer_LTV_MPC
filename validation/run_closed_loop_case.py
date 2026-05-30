"""Run one closed-loop Trailer LTV MPC validation case."""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trailer_ltv_mpc import load_controller_config

from validation.closed_loop import run_closed_loop_path
from validation.path_generation import direction_sign, make_validation_path


# Edit these defaults for quick interactive validation runs.
PATH_TYPE = "spline"  # "straight", "arc", "spline", or "harsh_turn"
DIRECTION = "reverse"  # "forward" or "reverse"
N_STEPS = 5000
PATH_SAMPLE_SPACING_M = 0.2
CONFIG_PATH = "configs/default.yaml"
SAVE_RESULTS = False
SHOW_FIGURES = True
SHOW_ANIMATION = True
ANIMATION_INTERVAL_MS = 15
ANIMATION_MAX_FRAMES = 800
SAVE_FIGURES = False
OUTPUT_DIR = "outputs/validation"


@dataclass(frozen=True)
class ValidationRunOptions:
    path: str = PATH_TYPE
    direction: str = DIRECTION
    steps: int = N_STEPS
    ds: float = PATH_SAMPLE_SPACING_M
    config: str = CONFIG_PATH
    save_results: bool = SAVE_RESULTS
    show_figures: bool = SHOW_FIGURES
    show_animation: bool = SHOW_ANIMATION
    animation_interval_ms: int = ANIMATION_INTERVAL_MS
    animation_max_frames: int | None = ANIMATION_MAX_FRAMES
    save_figures: bool = SAVE_FIGURES
    output_dir: str = OUTPUT_DIR


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_controller_config(args.config)
    result = run_case(args, config)
    summary = result.summary()
    print_summary(summary)

    output_dir = None
    if args.save_results or args.save_figures:
        output_dir = make_output_dir(result, Path(args.output_dir))
    if args.save_results:
        save_result(result, output_dir)
        print(f"Saved validation result: {output_dir}")
    if args.show_figures or args.show_animation or args.save_figures:
        render_visuals(result, config, args, output_dir)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=["straight", "arc", "spline", "harsh_turn"], default=PATH_TYPE)
    parser.add_argument("--direction", choices=["forward", "reverse"], default=DIRECTION)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--ds", type=float, default=PATH_SAMPLE_SPACING_M)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--save-results", action="store_true", default=SAVE_RESULTS)
    parser.add_argument("--show-figures", action=argparse.BooleanOptionalAction, default=SHOW_FIGURES)
    parser.add_argument("--show-animation", action=argparse.BooleanOptionalAction, default=SHOW_ANIMATION)
    parser.add_argument("--animation-interval-ms", type=int, default=ANIMATION_INTERVAL_MS)
    parser.add_argument("--animation-max-frames", type=int, default=ANIMATION_MAX_FRAMES)
    parser.add_argument("--save-figures", action="store_true", default=SAVE_FIGURES)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return ValidationRunOptions(**vars(parser.parse_args(argv)))


def run_case(args, config=None):
    if config is None:
        config = load_controller_config(args.config)
    if args.path == "harsh_turn" and args.direction == "reverse" and not config.enable_forward_correction:
        config = replace(config, enable_forward_correction=True)
    path = make_validation_path(args.path, args.direction, args.ds)
    metadata = {
        "path_kind": args.path,
        "direction": args.direction,
        "steps": args.steps,
        "ds": args.ds,
        "config_source": args.config,
        "enable_forward_correction": config.enable_forward_correction,
    }
    return run_closed_loop_path(path, direction_sign(args.direction), config, args.steps, metadata)


def render_visuals(result, config, args, output_dir=None):
    from validation.plotting import animate_tracking, plot_diagnostics, save_figures, show_visuals

    figures = None
    if args.show_figures or args.save_figures:
        figures = plot_diagnostics(result, config)
    if args.save_figures:
        save_figures(figures, output_dir)
        print(f"Saved validation figures: {output_dir}")
    animation = None
    if args.show_animation:
        animation = animate_tracking(
            result,
            config,
            interval_ms=args.animation_interval_ms,
            max_frames=args.animation_max_frames,
        )
    if args.show_figures or args.show_animation:
        show_visuals()
    return figures, animation


def print_summary(summary: dict):
    print("\nTrailer LTV MPC closed-loop validation")
    print(f"  case: {summary['path_kind']}_{summary['direction']}")
    print(f"  steps: {summary['steps']}")
    print(f"  config: {summary['config_source']}")
    print(f"  path length: {summary['path_length_m']:.3f} m")
    print(
        f"  final progress: {summary['final_progress_m']:.3f} m "
        f"({100.0 * summary['final_progress_fraction']:.1f}%)"
    )
    print(f"  reached end: {summary['reached_end']}")
    if summary["first_reached_end_time_s"] is not None:
        print(f"  first reached end: {summary['first_reached_end_time_s']:.2f} s")
    print(f"  max tracking error: {summary['max_error_m']:.3f} m")
    print(f"  terminal tracking error: {summary['terminal_error_m']:.3f} m")
    print(f"  max heading error: {np.rad2deg(summary['max_heading_error_rad']):.3f} deg")
    print(f"  max gamma: {np.rad2deg(summary['max_gamma_rad']):.3f} deg")
    print(
        "  solver: "
        f"{'solved all MPC steps' if summary['solver_succeeded'] else 'non-MPC or failed step present'}, "
        f"max iter {summary['max_solver_iterations']}, "
        f"mean iter {summary['mean_solver_iterations']:.1f}"
    )
    print(
        "  speed tracking: "
        f"cruise mean |V2_actual| {summary['mean_abs_cruise_V2_actual_mps']:.3f} m/s, "
        f"cruise mean |V2_ref_profile| {summary['mean_abs_cruise_V2_profile_ref_mps']:.3f} m/s, "
        f"max cruise error {summary['max_speed_reference_error_mps']:.3f} m/s"
    )
    print(
        "  validation checks: "
        f"end={'PASS' if summary['reached_end_ok'] else 'FAIL'}, "
        f"lateral={'PASS' if summary['lateral_error_ok'] else 'FAIL'}, "
        f"heading={'PASS' if summary['heading_error_ok'] else 'FAIL'}, "
        f"solver={'PASS' if summary['solver_succeeded'] else 'FAIL'}, "
        f"speed={'PASS' if summary['speed_reference_ok'] else 'FAIL'}"
    )
    print(f"  final repo state [X2, Y2, psi1, psi2]: {summary['final_repo_state']}")


def make_output_dir(result, output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_name = f"{result.metadata['path_kind']}_{result.metadata['direction']}"
    output_dir = output_root / f"{timestamp}_{case_name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_result(result, output_dir: Path) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary(), handle, indent=2)

    np.savez(
        output_dir / "timeseries.npz",
        step_idx=result.step_idx,
        t_s=result.t_s,
        repo_state=result.repo_state,
        truck_rear_x=result.truck_rear_x,
        truck_rear_y=result.truck_rear_y,
        hitch_x=result.hitch_x,
        hitch_y=result.hitch_y,
        stations_m=result.stations_m,
        errors_m=result.errors_m,
        heading_errors_rad=result.heading_errors_rad,
        gamma_rad=result.gamma_rad,
        delta_f=result.delta_f,
        V1=result.V1,
        delta_T=result.delta_T,
        delta_T_actual=result.delta_T_actual,
        V2=result.V2,
        V2_actual=result.V2_actual,
        V2_ref=result.V2_ref,
        V2_profile_ref=result.V2_profile_ref,
        V2_profile_start_scale=result.V2_profile_start_scale,
        V2_profile_end_scale=result.V2_profile_end_scale,
        V2_profile_combined_scale=result.V2_profile_combined_scale,
        solver_status=result.solver_status,
        solver_iterations=result.solver_iterations,
        search_start_idx=result.search_start_idx,
        reference_x=result.reference_x,
        reference_y=result.reference_y,
        reference_theta=result.reference_theta,
        reference_s=result.reference_s,
        reference_direction=result.reference_direction,
        mode=result.mode,
        phase=result.phase,
        correction_anchor_x=result.correction_anchor_x,
        correction_anchor_y=result.correction_anchor_y,
        correction_target_x=result.correction_target_x,
        correction_target_y=result.correction_target_y,
    )


if __name__ == "__main__":
    raise SystemExit(main())
