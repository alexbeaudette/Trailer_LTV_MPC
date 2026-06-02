"""MATLAB-style plotting helpers for closed-loop validation runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trailer_ltv_mpc import TrailerLtvMpcConfig
from trailer_ltv_mpc.geometry import measurement_from_repo_state

from validation.closed_loop import ClosedLoopResult


@dataclass(frozen=True)
class ValidationFigures:
    path_tracking: object
    tracking_errors: object
    steering: object
    motion: object

    def as_dict(self) -> dict[str, object]:
        return {
            "path_tracking": self.path_tracking,
            "tracking_errors": self.tracking_errors,
            "steering": self.steering,
            "motion": self.motion,
        }


def plot_path_tracking(result: ClosedLoopResult):
    """Plot reference and actual trailer-rear paths."""
    plt = _pyplot()
    fig, ax = plt.subplots(num="Trailer LTV MPC Path Tracking", figsize=(10.5, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        result.reference_x,
        result.reference_y,
        "--",
        color=(0.45, 0.45, 0.45),
        linewidth=3.0,
        label="Reference Trailer Path",
    )
    ax.plot(
        result.repo_state[:, 0],
        result.repo_state[:, 1],
        "-",
        color=(0.0, 0.30, 0.85),
        linewidth=3.0,
        label="Actual Trailer Path",
    )
    ax.set_title("Trailer-Only LTV MPC (FULL Plant) Trailer Path Tracking", fontsize=18, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    _style_single_axis(ax)
    _style_legend(ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.92), borderaxespad=0.0))
    fig.tight_layout()
    return fig


def plot_diagnostics(result: ClosedLoopResult, config: TrailerLtvMpcConfig) -> ValidationFigures:
    """Plot tracking, steering, and motion diagnostics."""
    tracking_errors = _plot_tracking_errors(result)
    steering = _plot_steering(result, config)
    motion = _plot_motion(result, config)
    path_tracking = plot_path_tracking(result)
    return ValidationFigures(
        path_tracking=path_tracking,
        tracking_errors=tracking_errors,
        steering=steering,
        motion=motion,
    )


def animate_tracking(
    result: ClosedLoopResult,
    config: TrailerLtvMpcConfig,
    *,
    interval_ms: int = 15,
    max_frames: int | None = 800,
):
    """Create a MATLAB-style truck-trailer tracking animation."""
    plt = _pyplot()
    animation = _animation_module()
    frames = _animation_frames(result, max_frames)
    body = _body_dimensions(config)

    fig, ax = plt.subplots(num="Trailer LTV MPC Animation", figsize=(14.0, 7.0))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    (reference_line,) = ax.plot(
        result.reference_x,
        result.reference_y,
        ":",
        color=(0.74, 0.74, 0.74),
        linewidth=2.8,
        label="Reference Path",
    )
    (trace_line,) = ax.plot([], [], "-", color=(0.22, 0.70, 1.0), linewidth=3.0, label="Trailer Rear Axle Path")
    (hitch_line,) = ax.plot([], [], "-", color=(0.22, 0.70, 1.0), linewidth=1.6, alpha=0.65)
    (hitch_marker,) = ax.plot([], [], "o", markerfacecolor=(1.0, 0.82, 0.25), markeredgecolor="white", markersize=6, markeredgewidth=1.2)
    (correction_line,) = ax.plot([], [], ":", color=(0.10, 0.85, 0.25), linewidth=2.5)
    (correction_anchor_marker,) = ax.plot(
        [],
        [],
        "o",
        markerfacecolor="white",
        markeredgecolor=(1.0, 0.10, 0.10),
        markersize=9,
        markeredgewidth=2.2,
        label="Forward correction path point",
    )
    (correction_target_marker,) = ax.plot(
        [],
        [],
        "o",
        markerfacecolor="white",
        markeredgecolor=(0.0, 0.85, 0.25),
        markersize=10,
        markeredgewidth=2.4,
        label="Forward correction target",
    )
    trailer_patch = plt.Polygon(
        np.zeros((4, 2)),
        closed=True,
        facecolor=(0.72, 0.72, 0.72, 0.42),
        edgecolor=(0.88, 0.88, 0.88),
        linewidth=2.0,
        label="Trailer Body",
    )
    truck_patch = plt.Polygon(
        np.zeros((4, 2)),
        closed=True,
        facecolor=(0.12, 0.72, 0.86, 0.38),
        edgecolor=(0.44, 0.92, 1.0),
        linewidth=2.0,
        label="Truck Body",
    )
    ax.add_patch(trailer_patch)
    ax.add_patch(truck_patch)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    _style_dark_axis(ax)
    _set_forward_correction_limits(ax, result, body)
    legend_handles = [reference_line, trace_line, trailer_patch, truck_patch]
    if _has_forward_correction_points(result):
        legend_handles.extend([correction_anchor_marker, correction_target_marker])
    _style_legend(
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.76),
            borderaxespad=0.0,
        )
    )
    fig.subplots_adjust(left=0.055, right=0.845, bottom=0.105, top=0.9)

    def update(frame_idx: int):
        measurement = measurement_from_repo_state(result.repo_state[frame_idx, :], config.geom)
        sample_idx = min(frame_idx, result.t_s.size - 1)

        trace_line.set_data(result.repo_state[: frame_idx + 1, 0], result.repo_state[: frame_idx + 1, 1])
        hitch_line.set_data([measurement.X1, measurement.Xh, measurement.X2], [measurement.Y1, measurement.Yh, measurement.Y2])
        hitch_marker.set_data([measurement.Xh], [measurement.Yh])
        _set_optional_marker(
            correction_anchor_marker,
            result.correction_anchor_x[sample_idx],
            result.correction_anchor_y[sample_idx],
        )
        _set_optional_marker(
            correction_target_marker,
            result.correction_target_x[sample_idx],
            result.correction_target_y[sample_idx],
        )
        _set_optional_line(
            correction_line,
            result.correction_anchor_x[sample_idx],
            result.correction_anchor_y[sample_idx],
            result.correction_target_x[sample_idx],
            result.correction_target_y[sample_idx],
        )
        trailer_polygon = _body_polygon(
            measurement.X2,
            measurement.Y2,
            measurement.psi2,
            body.trailer_front,
            body.trailer_rear,
            body.trailer_width,
        )
        truck_polygon = _body_polygon(
            measurement.X1,
            measurement.Y1,
            measurement.psi1,
            body.truck_front,
            body.truck_rear,
            body.truck_width,
        )
        trailer_patch.set_xy(trailer_polygon)
        truck_patch.set_xy(truck_polygon)
        ax.set_title(
            f"Trailer-Only LTV MPC (FULL Plant) Vehicle and Path Tracking | t = {sample_idx * config.Ts:.1f} s",
            fontsize=14,
            fontweight="bold",
            color="white",
        )
        return (
            trace_line,
            hitch_line,
            hitch_marker,
            correction_line,
            correction_anchor_marker,
            correction_target_marker,
            trailer_patch,
            truck_patch,
        )

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False, repeat=False)
    return anim


def animate_forward_correction_tracking(
    result: ClosedLoopResult,
    config: TrailerLtvMpcConfig,
    *,
    interval_ms: int = 15,
    max_frames: int | None = 800,
):
    """Create a forward-correction-focused truck-trailer tracking animation."""
    plt = _pyplot()
    animation = _animation_module()
    frames = _animation_frames(result, max_frames)
    body = _body_dimensions(config)

    fig, ax = plt.subplots(num="Trailer LTV MPC Forward Correction", figsize=(14.0, 7.0))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    (reference_line,) = ax.plot(
        result.reference_x,
        result.reference_y,
        "--",
        color=(0.70, 0.70, 0.70),
        linewidth=2.6,
        label="Reference Path",
    )
    (trace_line,) = ax.plot([], [], "-", color=(0.22, 0.70, 1.0), linewidth=3.0, label="Trailer Path")
    (hitch_marker,) = ax.plot([], [], "x", color=(1.0, 0.82, 0.25), markersize=8, markeredgewidth=2.0)
    (rear_marker,) = ax.plot([], [], "x", color=(0.22, 0.70, 1.0), markersize=8, markeredgewidth=2.0)
    (anchor_line,) = ax.plot([], [], ":", color=(0.10, 0.85, 0.25), linewidth=2.5)
    (aim_line,) = ax.plot([], [], ":", color=(0.65, 0.65, 0.65), linewidth=1.8, visible=False)
    (anchor_marker,) = ax.plot(
        [],
        [],
        "o",
        markerfacecolor="white",
        markeredgecolor=(1.0, 0.10, 0.10),
        markersize=9,
        markeredgewidth=2.2,
        label="Forward Correction Path Point",
    )
    (target_marker,) = ax.plot(
        [],
        [],
        "o",
        markerfacecolor="white",
        markeredgecolor=(0.0, 0.85, 0.25),
        markersize=10,
        markeredgewidth=2.4,
        label="Forward Correction Target",
    )
    trailer_patch = plt.Polygon(
        np.zeros((4, 2)),
        closed=True,
        facecolor=(0.72, 0.72, 0.72, 0.42),
        edgecolor=(0.88, 0.88, 0.88),
        linewidth=2.0,
        label="Trailer Body",
    )
    truck_patch = plt.Polygon(
        np.zeros((4, 2)),
        closed=True,
        facecolor=(0.12, 0.72, 0.86, 0.38),
        edgecolor=(0.44, 0.92, 1.0),
        linewidth=2.0,
        label="Truck Body",
    )
    ax.add_patch(trailer_patch)
    ax.add_patch(truck_patch)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    _style_dark_axis(ax)
    _set_forward_correction_limits(ax, result, body)
    _style_legend(
        ax.legend(
            handles=[reference_line, trace_line, trailer_patch, truck_patch, anchor_marker, target_marker],
            loc="center left",
            bbox_to_anchor=(1.01, 0.72),
            borderaxespad=0.0,
        )
    )
    fig.subplots_adjust(left=0.055, right=0.815, bottom=0.105, top=0.9)

    def update(frame_idx: int):
        sample_idx = min(frame_idx, result.t_s.size - 1)
        measurement = measurement_from_repo_state(result.repo_state[frame_idx, :], config.geom)

        trace_line.set_data(result.repo_state[: frame_idx + 1, 0], result.repo_state[: frame_idx + 1, 1])
        hitch_marker.set_data([measurement.Xh], [measurement.Yh])
        rear_marker.set_data([measurement.X2], [measurement.Y2])
        trailer_patch.set_xy(
            _body_polygon(
                measurement.X2,
                measurement.Y2,
                measurement.psi2,
                body.trailer_front,
                body.trailer_rear,
                body.trailer_width,
            )
        )
        truck_patch.set_xy(
            _body_polygon(
                measurement.X1,
                measurement.Y1,
                measurement.psi1,
                body.truck_front,
                body.truck_rear,
                body.truck_width,
            )
        )
        _set_optional_marker(anchor_marker, result.correction_anchor_x[sample_idx], result.correction_anchor_y[sample_idx])
        _set_optional_marker(target_marker, result.correction_target_x[sample_idx], result.correction_target_y[sample_idx])
        _set_optional_line(
            anchor_line,
            result.correction_anchor_x[sample_idx],
            result.correction_anchor_y[sample_idx],
            result.correction_target_x[sample_idx],
            result.correction_target_y[sample_idx],
        )
        _set_optional_line(
            aim_line,
            measurement.X1,
            measurement.Y1,
            result.correction_target_x[sample_idx],
            result.correction_target_y[sample_idx],
        )
        phase_text = ""
        if result.mode[sample_idx] == "forward_correction":
            phase_text = f" | {result.phase[sample_idx]}"
        ax.set_title(
            f"Forward Correction Tracking | k = {int(result.step_idx[sample_idx])}{phase_text}",
            fontsize=14,
            fontweight="bold",
            color="white",
        )
        return (
            trace_line,
            hitch_marker,
            rear_marker,
            anchor_line,
            aim_line,
            anchor_marker,
            target_marker,
            trailer_patch,
            truck_patch,
        )

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False, repeat=False)
    return anim


def plot_forward_correction_start_snapshot(
    result: ClosedLoopResult,
    config: TrailerLtvMpcConfig,
    *,
    case_title: str = "Forward Correction",
):
    """Plot the first sample where forward correction is active."""
    activation_idx = _first_forward_correction_activation_idx(result)
    plt = _pyplot()
    body = _body_dimensions(config)
    measurement = measurement_from_repo_state(result.repo_state[activation_idx, :], config.geom)
    closest_x = float(np.interp(result.stations_m[activation_idx], result.reference_s, result.reference_x))
    closest_y = float(np.interp(result.stations_m[activation_idx], result.reference_s, result.reference_y))

    fig, ax = plt.subplots(num="Forward Correction Start Snapshot", figsize=(11.5, 7.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(
        result.reference_x,
        result.reference_y,
        "-",
        color=(0.0, 0.30, 0.85),
        linewidth=2.8,
        label="Reference Path",
    )
    trailer_patch = plt.Polygon(
        _body_polygon(
            measurement.X2,
            measurement.Y2,
            measurement.psi2,
            body.trailer_front,
            body.trailer_rear,
            body.trailer_width,
        ),
        closed=True,
        facecolor=(0.62, 0.62, 0.62, 0.45),
        edgecolor=(0.20, 0.20, 0.20),
        linewidth=1.8,
        label="Trailer Body",
    )
    truck_patch = plt.Polygon(
        _body_polygon(
            measurement.X1,
            measurement.Y1,
            measurement.psi1,
            body.truck_front,
            body.truck_rear,
            body.truck_width,
        ),
        closed=True,
        facecolor=(0.0, 0.45, 0.95, 0.42),
        edgecolor=(0.0, 0.18, 0.50),
        linewidth=1.8,
        label="Truck Body",
    )
    ax.add_patch(trailer_patch)
    ax.add_patch(truck_patch)
    ax.plot(
        [measurement.X2],
        [measurement.Y2],
        "o",
        markerfacecolor="white",
        markeredgecolor=(0.0, 0.30, 0.85),
        markersize=8,
        markeredgewidth=2.0,
        label="Trailer Rear Axle",
    )
    ax.plot(
        [closest_x],
        [closest_y],
        "s",
        markerfacecolor="white",
        markeredgecolor=(0.0, 0.30, 0.85),
        markersize=8,
        markeredgewidth=2.0,
        label="Closest Path Point",
    )
    ax.plot(
        [result.correction_anchor_x[activation_idx], result.correction_target_x[activation_idx]],
        [result.correction_anchor_y[activation_idx], result.correction_target_y[activation_idx]],
        ":",
        color=(0.0, 0.65, 0.20),
        linewidth=2.4,
        label="Anchor Tangent Projection",
    )
    ax.plot(
        [result.correction_anchor_x[activation_idx]],
        [result.correction_anchor_y[activation_idx]],
        "o",
        markerfacecolor="white",
        markeredgecolor="red",
        markersize=9,
        markeredgewidth=2.2,
        label="Correction Anchor",
    )
    ax.plot(
        [result.correction_target_x[activation_idx]],
        [result.correction_target_y[activation_idx]],
        "o",
        markerfacecolor="white",
        markeredgecolor=(0.0, 0.65, 0.20),
        markersize=10,
        markeredgewidth=2.4,
        label="Correction Target",
    )
    ax.set_title(
        f"{case_title} | row {activation_idx} | t = {result.t_s[activation_idx]:.2f} s",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    _style_single_axis(ax)
    _set_forward_correction_limits(ax, result, body)
    _style_legend(ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.72), borderaxespad=0.0))
    fig.tight_layout()
    return fig


def save_figures(figures: ValidationFigures, output_dir) -> None:
    """Save validation figures as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.as_dict().items():
        fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")


def show_visuals() -> None:
    """Show all open Matplotlib figures and animations."""
    _pyplot().show()


def _plot_tracking_errors(result: ClosedLoopResult):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, num="Trailer LTV MPC Tracking Errors", figsize=(9.0, 6.0), sharex=True)
    _style_white_figure(fig, axes)
    axes[0].plot(result.t_s, result.errors_m, color=(0.0, 0.25, 0.95), linewidth=2.0)
    axes[0].set_ylabel("Lateral error [m]")
    axes[0].grid(True)
    axes[1].plot(result.t_s, np.rad2deg(result.heading_errors_rad), color=(0.85, 0.2, 0.1), linewidth=2.0)
    axes[1].set_ylabel("Heading error [deg]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True)
    fig.suptitle("Trailer LTV MPC Tracking Errors")
    fig.tight_layout()
    return fig


def _plot_steering(result: ClosedLoopResult, config: TrailerLtvMpcConfig):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, num="Trailer LTV MPC Steering Commands", figsize=(12.0, 7.0), sharex=True)
    _style_white_figure(fig, axes)
    axes[0].plot(result.t_s, np.rad2deg(result.delta_f), color=(0.10, 0.10, 0.10), linewidth=2.0, label="delta_f")
    axes[0].axhline(np.rad2deg(config.delta_f_max_rad), color=(0.8, 0.1, 0.1), linestyle="--", linewidth=1.4, label="Limit")
    axes[0].axhline(-np.rad2deg(config.delta_f_max_rad), color=(0.8, 0.1, 0.1), linestyle="--", linewidth=1.4)
    axes[0].set_ylabel("delta_f (deg)")
    _style_legend(axes[0].legend(loc="center right"))

    axes[1].plot(result.t_s, np.rad2deg(result.delta_T), "--", color=(0.0, 0.25, 0.95), linewidth=1.8, label="delta_T,cmd")
    axes[1].plot(result.t_s, np.rad2deg(result.delta_T_actual), color=(0.9, 0.15, 0.15), linewidth=2.0, label="delta_T,actual")
    axes[1].axhline(np.rad2deg(config.delta_T_min_rad), color=(0.8, 0.1, 0.1), linestyle="--", linewidth=1.2, label="delta_T,min")
    axes[1].axhline(np.rad2deg(config.delta_T_max_rad), color=(0.0, 0.25, 0.95), linestyle="--", linewidth=1.2, label="delta_T,max")
    axes[1].set_ylabel("delta_T (deg)")
    axes[1].set_xlabel("Time (s)")
    _style_legend(axes[1].legend(loc="lower right"))
    fig.suptitle("Trailer-Only LTV MPC (FULL Plant) Steering Commands", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def _plot_motion(result: ClosedLoopResult, config: TrailerLtvMpcConfig):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, num="Trailer LTV MPC Motion", figsize=(12.0, 6.5), sharex=True)
    _style_white_figure(fig, axes)
    axes[0].plot(result.t_s, np.rad2deg(result.gamma_rad), color=(0.10, 0.65, 0.25), linewidth=2.0, label="gamma")
    axes[0].set_ylabel("gamma (deg)")
    _style_legend(axes[0].legend(loc="center right"))

    axes[1].plot(result.t_s, result.V1, color=(0.0, 0.25, 0.95), linewidth=2.0, label="V1")
    axes[1].plot(result.t_s, result.V2_actual, color=(0.9, 0.35, 0.0), linewidth=2.0, label="V2")
    axes[1].plot(result.t_s, result.V2_ref, ":", color=(0.45, 0.45, 0.45), linewidth=1.5, label="V2 ref")
    axes[1].axhline(config.V2_max_abs_mps, color=(0.8, 0.1, 0.1), linestyle="--", linewidth=1.2, label="V2 bounds")
    axes[1].axhline(-config.V2_max_abs_mps, color=(0.8, 0.1, 0.1), linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_xlabel("Time (s)")
    _style_legend(axes[1].legend(loc="center right"))
    fig.suptitle("Trailer-Only LTV MPC (FULL Plant) Motion", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def _style_white_figure(fig, axes) -> None:
    fig.patch.set_facecolor("white")
    for ax in np.ravel(axes):
        _style_single_axis(ax)


def _style_single_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=(0.90, 0.90, 0.90), linewidth=0.8)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)


def _style_dark_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.grid(True, color=(0.25, 0.25, 0.25), linewidth=1.0, alpha=0.7)
    ax.tick_params(direction="in", top=True, right=True, colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color((0.74, 0.74, 0.74))
        spine.set_linewidth(1.2)


def _style_legend(legend) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor((0.55, 0.55, 0.55))
    for text in legend.get_texts():
        text.set_color("white")


@dataclass(frozen=True)
class _BodyDimensions:
    trailer_front: float
    trailer_rear: float
    trailer_width: float
    truck_front: float
    truck_rear: float
    truck_width: float
    follow_x_span: float
    follow_y_span: float


def _body_dimensions(config: TrailerLtvMpcConfig) -> _BodyDimensions:
    trailer_front = config.geom.L2
    trailer_rear = max(0.8, 0.10 * config.geom.L2)
    trailer_width = 2.6
    truck_front = config.geom.L1 + max(0.9, 0.30 * config.geom.L1)
    truck_rear = max(0.8, 0.25 * config.geom.L1)
    truck_width = 2.4
    total_rig_length = trailer_front + trailer_rear + truck_front + truck_rear
    follow_x_span = max(30.0, 3.0 * total_rig_length)
    follow_y_span = max(18.0, 2.5 * max(trailer_width, truck_width) + 10.0)
    return _BodyDimensions(
        trailer_front=trailer_front,
        trailer_rear=trailer_rear,
        trailer_width=trailer_width,
        truck_front=truck_front,
        truck_rear=truck_rear,
        truck_width=truck_width,
        follow_x_span=follow_x_span,
        follow_y_span=follow_y_span,
    )


def _set_forward_correction_limits(ax, result: ClosedLoopResult, body: _BodyDimensions) -> None:
    finite_anchor_x = result.correction_anchor_x[np.isfinite(result.correction_anchor_x)]
    finite_anchor_y = result.correction_anchor_y[np.isfinite(result.correction_anchor_y)]
    finite_target_x = result.correction_target_x[np.isfinite(result.correction_target_x)]
    finite_target_y = result.correction_target_y[np.isfinite(result.correction_target_y)]
    x_values = np.concatenate(
        [
            result.reference_x,
            result.repo_state[:, 0],
            result.truck_rear_x,
            result.hitch_x,
            finite_anchor_x,
            finite_target_x,
        ]
    )
    y_values = np.concatenate(
        [
            result.reference_y,
            result.repo_state[:, 1],
            result.truck_rear_y,
            result.hitch_y,
            finite_anchor_y,
            finite_target_y,
        ]
    )
    pad_x = max(4.0, 0.08 * (np.max(x_values) - np.min(x_values)), 0.5 * body.truck_front)
    pad_y = max(4.0, 0.08 * (np.max(y_values) - np.min(y_values)), 0.5 * body.truck_front)
    ax.set_xlim(float(np.min(x_values) - pad_x), float(np.max(x_values) + pad_x))
    ax.set_ylim(float(np.min(y_values) - pad_y), float(np.max(y_values) + pad_y))


def _set_optional_marker(marker, x_value: float, y_value: float) -> None:
    if np.isfinite(x_value) and np.isfinite(y_value):
        marker.set_data([x_value], [y_value])
    else:
        marker.set_data([], [])


def _set_optional_line(line, x0: float, y0: float, x1: float, y1: float) -> None:
    if np.all(np.isfinite([x0, y0, x1, y1])):
        line.set_data([x0, x1], [y0, y1])
    else:
        line.set_data([], [])


def _has_forward_correction_points(result: ClosedLoopResult) -> bool:
    return bool(
        np.any(
            np.isfinite(result.correction_anchor_x)
            & np.isfinite(result.correction_anchor_y)
            & np.isfinite(result.correction_target_x)
            & np.isfinite(result.correction_target_y)
        )
    )


def _first_forward_correction_activation_idx(result: ClosedLoopResult) -> int:
    active = (
        (result.mode == "forward_correction")
        & np.isfinite(result.correction_anchor_x)
        & np.isfinite(result.correction_anchor_y)
        & np.isfinite(result.correction_target_x)
        & np.isfinite(result.correction_target_y)
    )
    indices = np.flatnonzero(active)
    if not indices.size:
        raise ValueError("No forward-correction activation with finite anchor and target data was found.")
    return int(indices[0])


def _body_polygon(x_anchor: float, y_anchor: float, heading: float, front: float, rear: float, width: float) -> np.ndarray:
    longitudinal = np.array([np.cos(heading), np.sin(heading)])
    lateral = np.array([-np.sin(heading), np.cos(heading)])
    anchor = np.array([x_anchor, y_anchor], dtype=float)
    front_center = anchor + front * longitudinal
    rear_center = anchor - rear * longitudinal
    half_width = 0.5 * width
    return np.vstack(
        [
            front_center + half_width * lateral,
            front_center - half_width * lateral,
            rear_center - half_width * lateral,
            rear_center + half_width * lateral,
        ]
    )


def _animation_frames(result: ClosedLoopResult, max_frames: int | None) -> np.ndarray:
    frame_count = result.repo_state.shape[0]
    if max_frames is None or frame_count <= max_frames:
        return np.arange(frame_count)
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _animation_module():
    import matplotlib.animation as animation

    return animation
