# Validation Checklist

Run the lightest meaningful check before claiming success.

## Always Available

```bash
python tests/run_smoke.py
```

This should validate imports, deterministic geometry, mapping, speed profile,
reference construction, LTV model shapes, QP assembly shape, and pure-pursuit
forward correction.

## With Development Dependencies

```bash
pip install -e ".[dev]"
pytest
```

## With Solver Dependencies

```bash
python examples/run_basic_cases.py
```

## What To Check

- Geometry reconstruction matches front-hitch equations.
- `gamma` remains `psi1 - psi2`.
- Mapping uses the direction-aware forward/reverse conventions.
- `PathReference` station arrays remain strictly increasing.
- Adaptive speed scale respects configured min/max speed.
- Start/end profile allows zero speed only during active profile stages.
- QP matrices and bounds have expected dimensions.
- Controller outputs finite `delta_f`, `V1`, `delta_T`, and `V2`.
- Forward correction pure pursuit outputs finite saturated `delta_f`.

## If A Dependency Is Missing

Say exactly which check could not run and why. Do not imply full validation
passed if SciPy, OSQP, or pytest were unavailable.

