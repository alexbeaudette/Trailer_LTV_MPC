# Trailer LTV MPC

Python port of the Trailer LTV MPC controller and validation checks. The import package is `trailer_ltv_mpc`: planner/demo code produces `PathReference` arrays, and the controller consumes those arrays.

The port preserves the MATLAB notation:

- `gamma = psi1 - psi2`
- explicit controller state `[X1, Y1, psi1, X2, Y2, psi2]`
- Trailer LTV MPC state `[X2, Y2, psi2]`
- Trailer LTV MPC input `[delta_T, V2]`
- physical truck command `[delta_f, V1]`

The first forward-correction strategy uses pure pursuit for the forward leg and Trailer LTV MPC for reverse-to-anchor.

Controller and vehicle defaults can be loaded from YAML with `trailer_ltv_mpc.load_controller_config`. Path, simulation, and validation-case YAML files are placeholders for future scenario work. ROS2 simulation and testing should live in a separate adapter repo that depends on this Python package.

## Quick Smoke Test

```bash
pip install -e ".[dev]"
python -m pytest
python examples/run_basic_cases.py
```

Indices in the Python API are zero-based.
