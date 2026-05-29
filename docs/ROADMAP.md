# Roadmap

This repo is the lean Python controller port. It intentionally avoids the
MATLAB repo's milestone experiment structure, but it still tracks staged
capabilities.

## Phase 01: Lean Controller Skeleton

Status: initial scaffold complete.

- Package metadata and importable `src/trailer_controller/` layout.
- Dataclasses for geometry, paths, config, commands, and outputs.
- Deterministic math helpers, geometry reconstruction, mapping, speed profile,
  reference construction, LTV model, QP assembly, and pure-pursuit forward
  correction.
- Dependency-free smoke test.

## Phase 02: Dependency And Solver Validation

Status: next.

- Install development dependencies with `pip install -e ".[dev]"`.
- Run `pytest`.
- Install/use SciPy and OSQP.
- Run `python examples/run_basic_cases.py`.
- Fix solver-interface issues without changing controller math or tuning.

## Phase 03: MATLAB Fixture Parity

Status: planned.

- Export small MATLAB fixtures from the source repo.
- Compare Python outputs against MATLAB for:
  - geometry reconstruction
  - path projection/interpolation
  - adaptive speed scaling
  - start/end profile
  - virtual-to-actual mapping
  - admissible `delta_T` bounds
  - LTV model matrices
  - QP matrices and first command
  - pure-pursuit forward correction

## Phase 04: Controller Integration Cases

Status: planned.

- Run forward and reverse tracking cases using planner-generated
  `PathReference` arrays.
- Validate straight, arc, and spline-like paths.
- Validate that reverse behavior is not inferred from forward behavior.
- Monitor finite commands, solver status, sign consistency, mapping
  denominator, steering saturation, and completion.

## Phase 05: Forward Correction Integration

Status: planned.

- Validate pure-pursuit forward correction in harsh reverse cases.
- Validate transition from forward correction back to reverse Trailer LTV MPC.
- Keep LTV-MPC forward correction as a future strategy behind the same
  supervisor interface.

## Future Work

- LTV-MPC forward leg for forward correction.
- Stronger real-time interface.
- Planner integration.
- Optional visualization outside the controller core.

