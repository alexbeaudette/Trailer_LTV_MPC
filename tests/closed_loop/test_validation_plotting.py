"""Tests for MATLAB-style validation plotting helpers."""

import importlib.util

import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from trailer_ltv_mpc import TrailerLtvMpcConfig, load_controller_config

from validation.plotting import (
    animate_forward_correction_tracking,
    animate_tracking,
    plot_diagnostics,
    plot_forward_correction_start_snapshot,
    plot_path_tracking,
    save_figures,
)
from validation.run_forward_correction_case import ForwardCorrectionRunOptions, run_case as run_forward_correction_case
from validation.run_closed_loop_case import ValidationRunOptions, run_case


@pytest.mark.skipif(importlib.util.find_spec("osqp") is None, reason="OSQP not installed")
def test_path_tracking_plot_returns_figure():
    config = load_controller_config("configs/default.yaml")
    result = _short_result(config)

    fig = plot_path_tracking(result)

    assert fig.axes
    plt.close(fig)


@pytest.mark.skipif(importlib.util.find_spec("osqp") is None, reason="OSQP not installed")
def test_diagnostics_plots_return_figures():
    config = load_controller_config("configs/default.yaml")
    result = _short_result(config)

    figures = plot_diagnostics(result, config)

    assert set(figures.as_dict()) == {"path_tracking", "tracking_errors", "steering", "motion"}
    for fig in figures.as_dict().values():
        assert fig.axes
        plt.close(fig)


@pytest.mark.skipif(importlib.util.find_spec("osqp") is None, reason="OSQP not installed")
def test_animation_can_be_constructed():
    config = load_controller_config("configs/default.yaml")
    result = _short_result(config)

    animation = animate_tracking(result, config, max_frames=3)

    assert isinstance(animation, FuncAnimation)
    animation._draw_was_started = True
    plt.close(animation._fig)


@pytest.mark.skipif(importlib.util.find_spec("osqp") is None, reason="OSQP not installed")
def test_save_figures_writes_png_files(tmp_path):
    config = load_controller_config("configs/default.yaml")
    result = _short_result(config)
    figures = plot_diagnostics(result, config)

    save_figures(figures, tmp_path)

    assert (tmp_path / "path_tracking.png").is_file()
    assert (tmp_path / "tracking_errors.png").is_file()
    assert (tmp_path / "steering.png").is_file()
    assert (tmp_path / "motion.png").is_file()
    for fig in figures.as_dict().values():
        plt.close(fig)


def test_forward_correction_animation_can_be_constructed():
    config = TrailerLtvMpcConfig(enable_forward_correction=True, forward_correction_gamma_trigger_rad=0.0)
    result = _short_forward_correction_result(config)

    animation = animate_forward_correction_tracking(result, config, max_frames=3)

    assert isinstance(animation, FuncAnimation)
    animation._draw_was_started = True
    plt.close(animation._fig)


def test_forward_correction_start_snapshot_returns_figure():
    config = TrailerLtvMpcConfig(enable_forward_correction=True, forward_correction_gamma_trigger_rad=0.0)
    result = _short_forward_correction_result(config)

    fig = plot_forward_correction_start_snapshot(result, config)

    assert fig.axes
    plt.close(fig)


def test_forward_correction_start_snapshot_requires_activation():
    config = load_controller_config("configs/default.yaml")
    result = _short_result(config)

    with pytest.raises(ValueError, match="No forward-correction activation"):
        plot_forward_correction_start_snapshot(result, config)


def _short_result(config):
    args = ValidationRunOptions(
        path="straight",
        direction="forward",
        steps=5,
        ds=0.2,
        config="configs/default.yaml",
        save_results=False,
        show_figures=False,
        show_animation=False,
        save_figures=False,
        output_dir="outputs/validation",
    )
    return run_case(args, config)


def _short_forward_correction_result(config):
    args = ForwardCorrectionRunOptions(
        steps=4,
        ds=0.2,
        config="configs/default.yaml",
        show_animation=False,
        show_snapshot=False,
        save_figures=False,
        output_dir="outputs/forward_correction_validation",
    )
    return run_forward_correction_case(args, config)
