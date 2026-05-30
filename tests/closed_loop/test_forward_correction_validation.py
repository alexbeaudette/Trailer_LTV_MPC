"""Tests for the forward-correction validation script."""

from pathlib import Path

import numpy as np

from trailer_ltv_mpc import TrailerLtvMpcConfig
from validation.run_forward_correction_case import ForwardCorrectionRunOptions, main, run_case


def test_forward_correction_run_case_forces_forward_correction_enabled():
    config = TrailerLtvMpcConfig(enable_forward_correction=False, forward_correction_gamma_trigger_rad=0.0)
    args = ForwardCorrectionRunOptions(
        steps=3,
        ds=0.2,
        config="configs/default.yaml",
        show_animation=False,
        show_snapshot=False,
        output_dir="outputs/forward_correction_validation",
    )

    result = run_case(args, config)

    assert result.metadata["validation_case"] == "forward_correction"
    assert result.metadata["enable_forward_correction"]


def test_forward_correction_run_case_enters_mode_and_logs_target():
    config = TrailerLtvMpcConfig(enable_forward_correction=False, forward_correction_gamma_trigger_rad=0.0)
    args = ForwardCorrectionRunOptions(
        steps=4,
        ds=0.2,
        config="configs/default.yaml",
        show_animation=False,
        show_snapshot=False,
        output_dir="outputs/forward_correction_validation",
    )

    result = run_case(args, config)

    active = result.mode == "forward_correction"
    finite_target = (
        np.isfinite(result.correction_anchor_x)
        & np.isfinite(result.correction_anchor_y)
        & np.isfinite(result.correction_target_x)
        & np.isfinite(result.correction_target_y)
    )
    assert np.any(active)
    assert np.any(active & finite_target)


def test_forward_correction_cli_does_not_write_without_save_options(tmp_path: Path):
    exit_code = main(
        [
            "--steps",
            "3",
            "--no-show-animation",
            "--no-show-snapshot",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert not any(tmp_path.iterdir())
