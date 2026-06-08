"""Compare saved MATLAB spline runs against Python closed-loop validation logs."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MATLAB_CSV = Path("validation/matlab_runs.csv")
DEFAULT_PYTHON_OUTPUT_ROOT = Path("outputs/validation/matlab_compare")


@dataclass(frozen=True)
class MatlabCase:
    direction: str
    time_s: np.ndarray
    station_m: np.ndarray
    V2_runtime_mps: np.ndarray
    V2_cmd_mps: np.ndarray
    V2_profile_ref_mps: np.ndarray
    V2_profile_combined_scale: np.ndarray
    position_error_m: np.ndarray
    heading_error_rad: np.ndarray


@dataclass(frozen=True)
class PythonCase:
    direction: str
    time_s: np.ndarray
    station_m: np.ndarray
    V2_actual_mps: np.ndarray
    V2_cmd_mps: np.ndarray
    V2_profile_ref_mps: np.ndarray
    V2_profile_combined_scale: np.ndarray
    position_error_m: np.ndarray
    heading_error_rad: np.ndarray


def main(argv=None) -> int:
    args = parse_args(argv)
    matlab_cases = load_matlab_spline_cases(Path(args.matlab_csv))
    python_cases = {
        direction: load_latest_python_case(Path(args.python_output_root), direction)
        for direction in ("forward", "reverse")
    }

    for direction in ("forward", "reverse"):
        print_case_comparison(matlab_cases[direction], python_cases[direction])
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-csv", default=str(DEFAULT_MATLAB_CSV))
    parser.add_argument("--python-output-root", default=str(DEFAULT_PYTHON_OUTPUT_ROOT))
    return parser.parse_args(argv)


def load_matlab_spline_cases(path: Path) -> dict[str, MatlabCase]:
    rows_by_direction = {"forward": [], "reverse": []}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["path_type"] != "spline":
                continue
            direction = row["direction"]
            if direction not in rows_by_direction:
                continue
            if row["ref_mode"] == "initial":
                continue
            rows_by_direction[direction].append(row)

    return {
        direction: MatlabCase(
            direction=direction,
            time_s=_column(rows, "time_s"),
            station_m=_column(rows, "path_s_ref_m"),
            V2_runtime_mps=_column(rows, "V2_runtime_mps"),
            V2_cmd_mps=_column(rows, "V2_cmd_mps"),
            V2_profile_ref_mps=_column(rows, "V2_profile_ref_mps"),
            V2_profile_combined_scale=_column(rows, "V2_profile_combined_scale"),
            position_error_m=_column(rows, "position_error_m"),
            heading_error_rad=_column(rows, "heading_error_rad"),
        )
        for direction, rows in rows_by_direction.items()
    }


def load_latest_python_case(output_root: Path, direction: str) -> PythonCase:
    candidates = sorted(output_root.glob(f"*_spline_{direction}/timeseries.npz"))
    if not candidates:
        raise FileNotFoundError(f"No Python saved result found for spline {direction} under {output_root}.")
    path = candidates[-1]
    with np.load(path, allow_pickle=True) as data:
        return PythonCase(
            direction=direction,
            time_s=np.asarray(data["t_s"], dtype=float),
            station_m=np.asarray(data["stations_m"], dtype=float),
            V2_actual_mps=np.asarray(data["V2_actual"], dtype=float),
            V2_cmd_mps=np.asarray(data["V2"], dtype=float),
            V2_profile_ref_mps=np.asarray(data["V2_profile_ref"], dtype=float),
            V2_profile_combined_scale=np.asarray(data["V2_profile_combined_scale"], dtype=float),
            position_error_m=np.asarray(data["errors_m"], dtype=float),
            heading_error_rad=np.asarray(data["heading_errors_rad"], dtype=float),
        )


def print_case_comparison(matlab: MatlabCase, python: PythonCase) -> None:
    print(f"\nSpline {matlab.direction}")
    print(
        "  duration: "
        f"MATLAB {matlab.time_s[-1]:.2f} s, Python {python.time_s[-1]:.2f} s"
    )
    print(
        "  final station: "
        f"MATLAB {matlab.station_m[-1]:.2f} m, Python {python.station_m[-1]:.2f} m"
    )
    _print_signal_summary("MATLAB V2_runtime", matlab.V2_runtime_mps, matlab.V2_profile_ref_mps)
    _print_signal_summary("Python V2_actual", python.V2_actual_mps, python.V2_profile_ref_mps)
    _print_alignment_summary("time", matlab.time_s, matlab, python.time_s, python)
    _print_alignment_summary("station", matlab.station_m, matlab, python.station_m, python)
    print(
        "  max tracking error: "
        f"MATLAB {np.nanmax(matlab.position_error_m):.3f} m, "
        f"Python {np.nanmax(python.position_error_m):.3f} m"
    )
    print(
        "  max heading error: "
        f"MATLAB {np.rad2deg(np.nanmax(np.abs(matlab.heading_error_rad))):.3f} deg, "
        f"Python {np.rad2deg(np.nanmax(np.abs(python.heading_error_rad))):.3f} deg"
    )


def _print_signal_summary(label: str, actual: np.ndarray, profile: np.ndarray) -> None:
    print(
        f"  {label}: mean {np.nanmean(actual):.3f} m/s, "
        f"mean |.| {np.nanmean(np.abs(actual)):.3f} m/s, "
        f"profile mean |.| {np.nanmean(np.abs(profile)):.3f} m/s"
    )


def _print_alignment_summary(
    axis_name: str,
    matlab_axis: np.ndarray,
    matlab: MatlabCase,
    python_axis: np.ndarray,
    python: PythonCase,
) -> None:
    lo = max(float(np.nanmin(matlab_axis)), float(np.nanmin(python_axis)))
    hi = min(float(np.nanmax(matlab_axis)), float(np.nanmax(python_axis)))
    mask = (python_axis >= lo) & (python_axis <= hi)
    if np.count_nonzero(mask) < 2:
        print(f"  overlap by {axis_name}: not enough samples")
        return

    target_axis = python_axis[mask]
    matlab_runtime = _interp_monotonic(matlab_axis, matlab.V2_runtime_mps, target_axis)
    matlab_profile = _interp_monotonic(matlab_axis, matlab.V2_profile_ref_mps, target_axis)
    python_actual = python.V2_actual_mps[mask]
    python_profile = python.V2_profile_ref_mps[mask]

    runtime_diff = python_actual - matlab_runtime
    profile_diff = python_profile - matlab_profile
    print(
        f"  overlap by {axis_name}: "
        f"{lo:.2f}..{hi:.2f}, "
        f"mean |V2 actual/runtime diff| {np.nanmean(np.abs(runtime_diff)):.3f} m/s, "
        f"max {np.nanmax(np.abs(runtime_diff)):.3f} m/s"
    )
    print(
        f"  overlap by {axis_name}: "
        f"mean |profile diff| {np.nanmean(np.abs(profile_diff)):.3f} m/s, "
        f"max {np.nanmax(np.abs(profile_diff)):.3f} m/s"
    )


def _interp_monotonic(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    x_unique, y_unique = _unique_mean_xy(x, y)
    return np.interp(query, x_unique, y_unique)


def _unique_mean_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    sums = np.zeros_like(unique_x, dtype=float)
    counts = np.zeros_like(unique_x, dtype=float)
    np.add.at(sums, inverse, y_sorted)
    np.add.at(counts, inverse, np.isfinite(y_sorted).astype(float))
    return unique_x, sums / np.maximum(counts, 1.0)


def _column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([_float_or_nan(row[name]) for row in rows], dtype=float)


def _float_or_nan(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return np.nan


if __name__ == "__main__":
    raise SystemExit(main())
