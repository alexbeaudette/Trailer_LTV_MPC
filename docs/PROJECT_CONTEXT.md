# Project Context

## Project Overview

This repo contains the lean Python port of the Trailer LTV MPC controller for a
front-hitch tractor-trailer system.

The goal is a readable, importable controller library that preserves the
working MATLAB controller behavior without copying the MATLAB repo's
experiment-script sprawl.

The controller core lives in `src/trailer_controller/`. It consumes
precomputed `PathReference` arrays from a planner or demo layer.

## Current Scope

The first Python version includes:

- Trailer LTV MPC for forward and reverse trailer-path tracking.
- Dual-input virtual trailer control with `u_T = [delta_T, V2]`.
- Adaptive `V2` reference speed.
- Distance-based start/end speed profile.
- Direction-aware virtual-to-actual mapping from `(delta_T, V2)` to
  `(delta_f, V1)`.
- Forward correction using pure pursuit for the forward leg.

Future work may add LTV-MPC-based forward correction, richer planners, fixture
parity against MATLAB, and tighter integration tests.

## Critical Assumptions

- This is a front-hitch tractor-trailer system.
- The hitch is in front of the truck rear axle.
- The trailer is the primary system.
- `gamma = psi1 - psi2`.
- `delta_T` is virtual trailer steering at the hitch.
- `delta_f` is physical truck front steering.
- `V1` is truck rear-axle speed.
- `V2` is trailer rear-axle speed.

## Non-Negotiable Rules

- Do not use hitch-behind equations.
- Do not change notation or state ordering.
- Do not assume forward and reverse motion are symmetric.
- Do not tune controller weights, constraints, horizons, rate limits, or
  lookahead distances unless explicitly requested.
- Do not put planner, plotting, or experiment-runner logic inside the
  controller core.

## Angle Handling Rules

- Use `atan2(y, x)` for angle recovery.
- Wrap state angles after integration/update.
- Compute angle differences first, then wrap.
- Do not wrap individual angles before subtracting them.
- Do not wrap angles inside `sin()` or `cos()`.

## Repo Structure

- `src/trailer_controller/`: importable controller package.
- `examples/`: small demo planners and runnable examples.
- `tests/`: unit tests, smoke tests, and future MATLAB parity fixtures.
- `docs/`: project context, roadmap, cleanup plan, and agent instructions.

## How Codex Should Work With This Repo

- Read `AGENTS.md` and this file first.
- Preserve the controller conventions before making changes.
- Prefer small, tested edits.
- Keep behavior changes separate from cleanup-only changes.
- If a dependency is missing, say so clearly rather than implying full
  validation passed.

