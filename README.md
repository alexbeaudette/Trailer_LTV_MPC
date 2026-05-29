# Lean Trailer Controller

Clean Python port of the Trailer LTV MPC controller. The package is an importable library: planner/demo code produces `PathReference` arrays, and the controller consumes those arrays.

The port preserves the MATLAB notation:

- `gamma = psi1 - psi2`
- explicit controller state `[X1, Y1, psi1, X2, Y2, psi2]`
- Trailer LTV MPC state `[X2, Y2, psi2]`
- Trailer LTV MPC input `[delta_T, V2]`
- physical truck command `[delta_f, V1]`

The first forward-correction strategy uses pure pursuit for the forward leg and Trailer LTV MPC for reverse-to-anchor.

## Quick Smoke Test

```bash
pip install -e ".[dev]"
pytest
python examples/run_basic_cases.py
```

Indices in the Python API are zero-based.
