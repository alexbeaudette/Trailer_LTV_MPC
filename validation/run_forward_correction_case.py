"""Run the reverse harsh-turn forward-correction validation case."""

import argparse
from collections import Counter
from dataclasses import dataclass, replace
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
from validation.run_closed_loop_case import make_output_dir, print_summary, save_result


PATH_TYPE = "harsh_turn"
DIRECTION = "reverse"
N_STEPS = 10000
PATH_SAMPLE_SPACING_M = 0.2
CONFIG_PATH = "configs/default.yaml"
SAVE_RESULTS = False
SHOW_ANIMATION = True
SHOW_SNAPSHOT = True
ANIMATION_INTERVAL_MS = 15
ANIMATION_MAX_FRAMES = 800
SAVE_FIGURES = False
OUTPUT_DIR = "outputs/forward_correction_validation"


@dataclass(frozen=True)
class ForwardCorrectionRunOptions:
    steps: int = N_STEPS
    ds: float = PATH_SAMPLE_SPACING_M
    config: str = CONFIG_PATH
    save_results: bool = SAVE_RESULTS
    show_animation: bool = SHOW_ANIMATION
    show_snapshot: bool = SHOW_SNAPSHOT
    animation_interval_ms: int = ANIMATION_INTERVAL_MS
    animation_max_frames: int | None = ANIMATION_MAX_FRAMES
    save_figures: bool = SAVE_FIGURES
    output_dir: str = OUTPUT_DIR


def main(argv=None) -> int:
    args = parse_args(argv)
    config = forward_correction_config(load_controller_config(args.config))
    result = run_case(args, config)
    print_summary(result.summary())
    print_forward_correction_summary(result)

    output_dir = None
    if args.save_results or args.save_figures:
        output_dir = make_output_dir(result, Path(args.output_dir))
    if args.save_results:
        save_result(result, output_dir)
        print(f"Saved validation result: {output_dir}")
    if args.show_animation or args.show_snapshot or args.save_figures:
        render_visuals(result, config, args, output_dir)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--ds", type=float, default=PATH_SAMPLE_SPACING_M)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--save-results", action="store_true", default=SAVE_RESULTS)
    parser.add_argument("--show-animation", action=argparse.BooleanOptionalAction, default=SHOW_ANIMATION)
    parser.add_argument("--show-snapshot", action=argparse.BooleanOptionalAction, default=SHOW_SNAPSHOT)
    parser.add_argument("--animation-interval-ms", type=int, default=ANIMATION_INTERVAL_MS)
    parser.add_argument("--animation-max-frames", type=int, default=ANIMATION_MAX_FRAMES)
    parser.add_argument("--save-figures", action="store_true", default=SAVE_FIGURES)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return ForwardCorrectionRunOptions(**vars(parser.parse_args(argv)))


def run_case(args: ForwardCorrectionRunOptions, config=None):
    if config is None:
        config = forward_correction_config(load_controller_config(args.config))
    else:
        config = forward_correction_config(config)
    path = make_validation_path(PATH_TYPE, DIRECTION, args.ds)
    metadata = {
        "path_kind": PATH_TYPE,
        "direction": DIRECTION,
        "steps": args.steps,
        "ds": args.ds,
        "config_source": args.config,
        "enable_forward_correction": config.enable_forward_correction,
        "validation_case": "forward_correction",
    }
    return run_closed_loop_path(path, direction_sign(DIRECTION), config, args.steps, metadata)


def forward_correction_config(config):
    return replace(config, enable_forward_correction=True)


def render_visuals(result, config, args, output_dir=None):
    from validation.plotting import (
        animate_forward_correction_tracking,
        plot_forward_correction_start_snapshot,
        show_visuals,
    )

    snapshot = None
    if args.show_snapshot or args.save_figures:
        snapshot = plot_forward_correction_start_snapshot(result, config, case_title="Reverse Harsh-Turn Forward Correction")
    animation = None
    if args.show_animation:
        animation = animate_forward_correction_tracking(
            result,
            config,
            interval_ms=args.animation_interval_ms,
            max_frames=args.animation_max_frames,
        )
    if args.save_figures:
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot.savefig(output_dir / "forward_correction_start_snapshot.png", dpi=150, bbox_inches="tight")
        print(f"Saved forward-correction figures: {output_dir}")
    if args.show_animation or args.show_snapshot:
        show_visuals()
    return snapshot, animation


def print_forward_correction_summary(result) -> None:
    active = result.mode == "forward_correction"
    active_indices = np.flatnonzero(active)
    finite_anchor_target = (
        np.isfinite(result.correction_anchor_x)
        & np.isfinite(result.correction_anchor_y)
        & np.isfinite(result.correction_target_x)
        & np.isfinite(result.correction_target_y)
    )
    first_time = result.t_s[active_indices[0]] if active_indices.size else None
    phase_counts = Counter(result.phase[active].tolist())

    print("\nForward-correction validation")
    if first_time is None:
        print("  first activation: none")
    else:
        print(f"  first activation: row {int(active_indices[0])}, t = {float(first_time):.2f} s")
    print(f"  active samples: {int(np.count_nonzero(active))}")
    print(f"  phase counts: {_format_counts(phase_counts)}")
    print(f"  finite anchor/target samples: {int(np.count_nonzero(active & finite_anchor_target))}")


def _format_counts(counts: Counter) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
