# Python Controller Conventions

## Purpose

This repo is a cleaned Python controller library, not a direct mirror of the
MATLAB research repo. Preserve the math and behavior, but do not recreate the
old script sprawl.

## Naming And State Order

- `gamma = psi1 - psi2`
- `delta_f`: physical truck steering.
- `delta_T`: virtual trailer steering.
- `V1`: truck rear-axle speed.
- `V2`: trailer rear-axle speed.
- Full plant state: `[X2, Y2, psi1, psi2]`.
- Explicit controller state: `[X1, Y1, psi1, X2, Y2, psi2]`.
- Trailer LTV MPC state: `[X2, Y2, psi2]`.
- Trailer LTV MPC input: `[delta_T, V2]`.

## Code Organization

- Controller logic lives in `src/trailer_ltv_mpc/`.
- Examples live in `examples/`.
- Tests live in `tests/`.
- Planner/demo code may create paths, but core controller code consumes
  `PathReference` arrays.
- Keep plotting out of `src/trailer_ltv_mpc/`.
- Keep solver-specific details behind the `QpSolver` adapter.

## Style

- Prefer dataclasses for structured data.
- Prefer pure functions for math/model helpers.
- Keep debug data readable, but do not let debug structures drive the public
  API.
- Avoid broad rewrites unless the current module boundary is actively blocking
  correctness or readability.
