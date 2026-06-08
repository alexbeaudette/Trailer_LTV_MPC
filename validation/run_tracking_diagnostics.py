"""Run path-tracking diagnostics for the standard validation path set."""

import argparse
from dataclasses import dataclass
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


PATH_TYPES = ("straight", "spline")
DIRECTIONS = ("forward", "reverse")
PATH_SAMPLE_SPACING_M = 0.2
CONFIG_PATH = "configs/default.yaml"
TIME_MARGIN_S = 45.0


@dataclass(frozen=True)
class DiagnosticCase:
    path: str
    direction: str
    steps: int


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_controller_config(args.config)
    cases = make_cases(args, config)
    failures = 0
    print(_header())
    for case in cases:
        try:
            path = make_validation_path(case.path, case.direction, args.ds)
            result = run_closed_loop_path(
                path,
                direction_sign(case.direction),
                config,
                case.steps,
                {
                    "path_kind": case.path,
                    "direction": case.direction,
                    "steps": case.steps,
                    "ds": args.ds,
                    "config_source": args.config,
                },
            )
            summary = result.summary()
            passed = all(
                [
                    summary["reached_end_ok"],
                    summary["lateral_error_ok"],
                    summary["heading_error_ok"],
                    summary["solver_succeeded"],
                    summary["speed_reference_ok"],
                ]
            )
            failures += 0 if passed else 1
            print(_format_row(case, summary, passed))
        except Exception as exc:
            failures += 1
            print(f"{case.path:8s} {case.direction:7s} ERROR {type(exc).__name__}: {exc}")
    return 1 if failures else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        choices=["straight", "arc", "sinusoid", "spline", "harsh_turn"],
        default=list(PATH_TYPES),
    )
    parser.add_argument("--directions", nargs="+", choices=["forward", "reverse"], default=list(DIRECTIONS))
    parser.add_argument("--steps", type=int, default=None, help="Override automatically sized step counts.")
    parser.add_argument("--ds", type=float, default=PATH_SAMPLE_SPACING_M)
    parser.add_argument("--config", default=CONFIG_PATH)
    return parser.parse_args(argv)


def make_cases(args, config):
    cases = []
    for path_kind in args.paths:
        for direction in args.directions:
            path = make_validation_path(path_kind, direction, args.ds)
            if args.steps is None:
                seconds = path.length_m / max(abs(config.V2_reference_mps), np.finfo(float).eps) + TIME_MARGIN_S
                steps = int(np.ceil(seconds / config.Ts))
            else:
                steps = args.steps
            cases.append(DiagnosticCase(path_kind, direction, steps))
    return cases


def _header():
    return (
        "path     dir     status steps  end_s   progress  max_e  max_head  "
        "speed_err  solver_iter"
    )


def _format_row(case, summary, passed):
    end_time = summary["first_reached_end_time_s"]
    end_text = f"{end_time:6.1f}" if end_time is not None else "  n/a "
    status = "PASS" if passed else "FAIL"
    return (
        f"{case.path:8s} {case.direction:7s} {status:5s} "
        f"{summary['steps']:5d} {end_text} "
        f"{summary['final_progress_m']:8.2f}/{summary['path_length_m']:.2f} "
        f"{summary['max_error_m']:6.3f} "
        f"{np.rad2deg(summary['max_heading_error_rad']):8.3f} "
        f"{summary['max_speed_reference_error_mps']:9.3f} "
        f"{summary['max_solver_iterations']:5d}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
