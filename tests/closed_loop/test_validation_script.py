"""Tests for the user-facing closed-loop validation script."""

from pathlib import Path

import numpy as np

from trailer_ltv_mpc import TrailerLtvMpcConfig
from validation.run_closed_loop_case import ValidationRunOptions, main, parse_args, run_case


def test_run_case_returns_plot_ready_result():
    args = ValidationRunOptions(
        path="straight",
        direction="forward",
        steps=8,
        ds=0.2,
        config="configs/default.yaml",
        save_results=False,
        output_dir="outputs/validation",
    )

    result = run_case(args)

    assert result.repo_state.shape == (9, 4)
    assert result.truck_rear_x.shape == (8,)
    assert result.hitch_x.shape == (8,)
    assert result.delta_f.shape == (8,)
    assert result.delta_T_actual.shape == (8,)
    assert result.V2_actual.shape == (8,)
    assert result.reference_x.size == result.reference_y.size
    assert np.all(np.isfinite(result.V2_profile_ref))


def test_script_defaults_are_usable():
    args = parse_args([])

    result = run_case(args)

    assert args.path in {"straight", "arc", "spline", "harsh_turn"}
    assert args.direction in {"forward", "reverse"}
    assert result.repo_state.shape[0] <= args.steps + 1
    assert result.metadata["requested_steps"] == args.steps


def test_run_case_stops_after_reaching_path_end():
    args = ValidationRunOptions(
        path="straight",
        direction="forward",
        steps=800,
        ds=0.2,
        config="configs/default.yaml",
        save_results=False,
        output_dir="outputs/validation",
    )

    result = run_case(args)

    assert result.reached_end
    assert result.metadata["terminated_at_path_end"]
    assert result.metadata["requested_steps"] == args.steps
    assert result.metadata["steps"] < args.steps
    assert result.repo_state.shape == (result.metadata["steps"] + 1, 4)
    assert result.delta_f.shape == (result.metadata["steps"],)


def test_cli_does_not_write_without_save_results(tmp_path: Path):
    exit_code = main(
        [
            "--path",
            "straight",
            "--direction",
            "forward",
            "--steps",
            "4",
            "--no-show-figures",
            "--no-show-animation",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert not any(tmp_path.iterdir())


def test_forward_correction_does_not_start_before_trigger():
    config = TrailerLtvMpcConfig(enable_forward_correction=True)
    args = ValidationRunOptions(
        path="straight",
        direction="reverse",
        steps=4,
        ds=0.2,
        config="configs/default.yaml",
        save_results=False,
        output_dir="outputs/validation",
    )

    result = run_case(args, config)

    assert np.all(result.mode == "trailer_ltv_mpc")
    assert np.all(result.phase == "tracking")
    assert np.all(np.isfinite(result.delta_T_actual))
    assert np.all(np.isfinite(result.V2_actual))
    assert np.all(np.isnan(result.correction_anchor_x))
    assert np.all(np.isnan(result.correction_target_x))
