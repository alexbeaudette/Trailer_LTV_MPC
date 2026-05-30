# Cleanup Plan

This document tracks cleanup work for the lean Python controller repo. Cleanup
must improve readability without changing controller behavior unless the user
explicitly asks for a behavior change.

## Pass 1: Orientation Layer

Status: done.

- Add `README.md`.
- Add `AGENTS.md`.
- Add Python-specific project context, roadmap, cleanup plan, agent docs, and
  agent prompts.
- Add package `.gitignore`.

## Pass 2: Dependency Setup

Status: pending.

Recommended cleanup:

- Confirm `pip install -e ".[dev]"` works in the new repo.
- Confirm `pytest` runs.
- Confirm SciPy and OSQP are installed and available.
- Keep optional/development dependencies in `pyproject.toml`, not hidden in
  scripts.

## Pass 3: Controller API Readability

Status: pending.

Recommended cleanup:

- Add or tighten top-of-file module docstrings so each source file immediately
  explains its role in the controller package.
- Review public dataclasses for naming consistency.
- Keep dataclass defaults simple and intentional. Use `default_factory` only for
  mutable defaults or other standard Python cases where direct defaults would be
  incorrect.
- Keep `ControllerOutput` and debug fields readable without bloating the public
  API.
- Remove opportunistic fallback/auto-repair logic that hides invalid inputs
  unless it is part of an explicitly documented controller contract.
- Confirm zero-based indexing is clearly documented for Python.
- Keep planner-generated arrays outside the controller core.

## Pass 4: MATLAB Parity Fixtures

Status: pending.

Recommended cleanup:

- Add a small fixture format under `tests/fixtures/`.
- Avoid dumping giant MATLAB result structs.
- Store only focused arrays needed for parity checks.
- Separate deterministic math tolerances from solver-output tolerances.

## Pass 5: Solver Isolation

Status: pending.

Recommended cleanup:

- Keep OSQP-specific setup in `qp_solver.py`.
- Keep QP matrix assembly testable without OSQP.
- Add tests for `H`, `f`, bounds, and rate constraints before relying on
  closed-loop behavior.

## Pass 6: Example Separation

Status: pending.

Recommended cleanup:

- Keep `examples/demo_planner.py` small.
- Do not let example planner logic drift into `src/trailer_ltv_mpc/`.
- Put future plotting or animation in examples or separate tools, not in the
  controller package.
