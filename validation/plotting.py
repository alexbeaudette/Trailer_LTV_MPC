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

    fig, ax = plt.subplots(num="Trailer LTV MPC Animation", figsize=(9.0, 6.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(
        result.reference_x,
        result.reference_y,
        "-",
        color=(0.0, 0.30, 0.85),
        linewidth=2.4,
        label="Reference Path",
    )
    (trace_line,) = ax.plot([], [], "-", color=(0.0, 0.45, 1.0), linewidth=2.0, label="Trailer trace")
    (hitch_line,) = ax.plot([], [], "-", color=(0.08, 0.08, 0.08), linewidth=1.8)
    (trailer_axle_line,) = ax.plot([], [], "-", color="black", linewidth=2.2)
    (truck_axle_line,) = ax.plot([], [], "-", color="black", linewidth=2.2)
    (rear_marker,) = ax.plot([], [], "o", markerfacecolor="black", markeredgecolor="black", markersize=5, label="Trailer Rear Axle")
    (hitch_marker,) = ax.plot([], [], "o", markerfacecolor="white", markeredgecolor="black", markersize=5, markeredgewidth=1.4)
    (truck_marker,) = ax.plot([], [], "o", markerfacecolor="black", markeredgecolor="black", markersize=5)
    (path_marker,) = ax.plot([], [], "s", markerfacecolor="white", markeredgecolor=(0.0, 0.30, 0.85), markersize=7, markeredgewidth=2.0, label="Closest Path Point")
    (anchor_line,) = ax.plot([], [], ":", color=(0.0, 0.65, 0.20), linewidth=2.2, label="Anchor Tangent Projection")
    (anchor_marker,) = ax.plot([], [], "o", markerfacecolor="white", markeredgecolor="red", markersize=9, markeredgewidth=2.2, label="Correction Anchor")
    (target_marker,) = ax.plot([], [], "o", markerfacecolor="white", markeredgecolor=(0.0, 0.65, 0.20), markersize=10, markeredgewidth=2.4, label="Correction Target")
    trailer_patch = plt.Polygon(
        np.zeros((4, 2)),
        closed=True,
        facecolor=(0.62, 0.62, 0.62),
        edgecolor=(0.05, 0.05, 0.05),
        linewidth=1.6,
        alpha=0.95,
        label="Trailer Body",
    )
    truck_patch = plt.Polygon(
        np.zeros((4, 2)),
        closed=True,
        facecolor=(0.0, 0.45, 0.95),
        edgecolor=(0.0, 0.12, 0.35),
        linewidth=1.6,
        alpha=0.95,
        label="Truck Body",
    )
    ax.add_patch(trailer_patch)
    ax.add_patch(truck_patch)
    wheel_patches = [
        plt.Circle((0.0, 0.0), 0.22, facecolor="black", edgecolor="black", zorder=6)
        for _ in range(8)
    ]
    for wheel in wheel_patches:
        ax.add_patch(wheel)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    _style_single_axis(ax)
    _set_animation_limits(ax, result, body)
    _style_legend(ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.82), borderaxespad=0.0))

    def update(frame_idx: int):
        measurement = measurement_from_repo_state(result.repo_state[frame_idx, :], config.geom)
        sample_idx = min(frame_idx, result.t_s.size - 1)

        trace_line.set_data(result.repo_state[: frame_idx + 1, 0], result.repo_state[: frame_idx + 1, 1])
        hitch_line.set_data([measurement.X1, measurement.Xh, measurement.X2], [measurement.Y1, measurement.Yh, measurement.Y2])
        rear_marker.set_data([measurement.X2], [measurement.Y2])
        hitch_marker.set_data([measurement.Xh], [measurement.Yh])
        truck_marker.set_data([measurement.X1], [measurement.Y1])
        path_marker.set_data([result.reference_x[result.search_start_idx[sample_idx]]], [result.reference_y[result.search_start_idx[sample_idx]]])
        _set_optional_marker(anchor_marker, result.correction_anchor_x[sample_idx], result.correction_anchor_y[sample_idx])
        _set_optional_marker(target_marker, result.correction_target_x[sample_idx], result.correction_target_y[sample_idx])
        _set_optional_line(
            anchor_line,
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
        _set_axle_line(trailer_axle_line, measurement.X2, measurement.Y2, measurement.psi2, body.trailer_width)
        _set_axle_line(truck_axle_line, measurement.X1, measurement.Y1, measurement.psi1, body.truck_width)
        _set_wheels(
            wheel_patches,
            [
                *_wheel_points(measurement.X2, measurement.Y2, measurement.psi2, body.trailer_rear, 0.72 * body.trailer_front, body.trailer_width),
                *_wheel_points(measurement.X1, measurement.Y1, measurement.psi1, body.truck_rear, 0.55 * body.truck_front, body.truck_width),
            ],
        )
        ax.set_title(
            f"Trailer-Only LTV MPC (FULL Plant) Vehicle and Path Tracking | t = {sample_idx * config.Ts:.1f} s",
            fontsize=14,
            fontweight="bold",
        )
        return (
            trace_line,
            hitch_line,
            trailer_axle_line,
            truck_axle_line,
            rear_marker,
            hitch_marker,
            truck_marker,
            path_marker,
            anchor_line,
            anchor_marker,
            target_marker,
            trailer_patch,
            truck_patch,
            *wheel_patches,
        )

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False, repeat=False)
    return anim


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


def _set_animation_limits(ax, result: ClosedLoopResult, body: _BodyDimensions) -> None:
    x_values = np.concatenate(
        [
            result.reference_x,
            result.repo_state[:, 0],
            result.truck_rear_x,
            result.hitch_x,
            result.correction_anchor_x[np.isfinite(result.correction_anchor_x)],
            result.correction_target_x[np.isfinite(result.correction_target_x)],
        ]
    )
    y_values = np.concatenate(
        [
            result.reference_y,
            result.repo_state[:, 1],
            result.truck_rear_y,
            result.hitch_y,
            result.correction_anchor_y[np.isfinite(result.correction_anchor_y)],
            result.correction_target_y[np.isfinite(result.correction_target_y)],
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


def _set_axle_line(line, x_anchor: float, y_anchor: float, heading: float, width: float) -> None:
    lateral = np.array([-np.sin(heading), np.cos(heading)])
    center = np.array([x_anchor, y_anchor], dtype=float)
    half_width = 0.56 * width
    ends = np.vstack([center - half_width * lateral, center + half_width * lateral])
    line.set_data(ends[:, 0], ends[:, 1])


def _wheel_points(x_anchor: float, y_anchor: float, heading: float, rear: float, front: float, width: float) -> list[np.ndarray]:
    longitudinal = np.array([np.cos(heading), np.sin(heading)])
    lateral = np.array([-np.sin(heading), np.cos(heading)])
    anchor = np.array([x_anchor, y_anchor], dtype=float)
    half_width = 0.56 * width
    axle_offsets = (-rear, front)
    return [
        anchor + axle_offset * longitudinal + side * half_width * lateral
        for axle_offset in axle_offsets
        for side in (-1.0, 1.0)
    ]


def _set_wheels(wheels, points: list[np.ndarray]) -> None:
    for wheel, point in zip(wheels, points):
        wheel.center = (float(point[0]), float(point[1]))


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
