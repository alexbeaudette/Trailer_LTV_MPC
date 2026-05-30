# AGENTS.md

## Project Overview

This repo contains the lean Python port of the Trailer LTV MPC controller for a
front-hitch tractor-trailer system.

The main artifact is an importable Python controller library under
`src/trailer_ltv_mpc/`. Examples and tests may use scripts, but controller
logic must stay separate from demos, plotting, simulation experiments, and
validation scaffolding.

## Critical Conventions

- This is a front-hitch tractor-trailer system. The hitch is in front of the
  truck rear axle.
- Preserve `gamma = psi1 - psi2`.
- Preserve notation:
  - `delta_f`: physical truck front steering angle.
  - `delta_T`: virtual trailer steering angle at the hitch.
  - `V1`: truck rear-axle speed.
  - `V2`: trailer rear-axle speed.
- Full plant state is `[X2, Y2, psi1, psi2]`.
- Explicit controller state is `[X1, Y1, psi1, X2, Y2, psi2]`.
- Trailer LTV MPC state is `[X2, Y2, psi2]`.
- Trailer LTV MPC input is `[delta_T, V2]`.
- Use `atan2(y, x)` for angle recovery.
- Compute angle differences first, then wrap.
- Do not wrap angles inside `sin()` or `cos()`.
- Do not tune weights, constraints, horizons, rate limits, or lookahead
  distances unless explicitly requested.

## Architecture Rules

- Keep the controller package importable and readable.
- The controller consumes `PathReference` arrays from a planner/demo layer.
- Do not put planner logic in the controller core.
- Do not put plotting in the controller core.
- Keep solver-specific code isolated in `qp_solver.py`.
- Keep forward-correction strategy logic isolated in `forward_correction.py`.
- Prefer explicit dataclasses and small functions over script-style modules.

## Readability Rules

- Keep the Python port lean. Do not add blanket fallback behavior, defensive
  branches, auto-repair logic, or "if blank, guess this" handling unless the
  caller contract explicitly requires it.
- Prefer clear inputs, explicit errors, and documented assumptions over hidden
  safety features at every function boundary.
- Use standard Python safety only when it prevents real language-level hazards,
  such as `default_factory` for mutable dataclass defaults.
- Every Python module must start with a short module docstring that explains
  what the file does at a high level. For non-Python files, add an equivalent
  top-of-file comment when the file format supports comments.

## Python Workflow

Before claiming success, run at least:

```bash
python tests/run_smoke.py
```

When dependencies are installed, also run:

```bash
pip install -e ".[dev]"
pytest
python examples/run_basic_cases.py
```

If SciPy, OSQP, or pytest are unavailable, say exactly which checks could not
run and which dependency is missing.

## Agent Docs

Read these before larger changes:

- `docs/agent_docs/gates.md`
- `docs/agent_docs/conventions.md`
- `docs/agent_docs/model_equations.md`
- `docs/agent_docs/validation_checklist.md`

Use prompts in `docs/agent_prompts/` when delegating or structuring work.
