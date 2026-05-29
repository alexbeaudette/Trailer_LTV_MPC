# Agent Gates

Use these gates for substantial work. Small typo fixes may skip straight to
implementation, but controller behavior changes should not.

## Gate 1: Analysis

Purpose: understand current behavior without editing.

Required output:

1. Files inspected
2. Relevant equations or logic found
3. Confirmed issues
4. Possible issues
5. Minimal recommended change
6. Validation required

## Gate 2: Planning

Purpose: convert the goal into a decision-complete implementation plan.

Rules:

- Do not edit code.
- Preserve notation, state ordering, and front-hitch conventions.
- Split controller/model work from examples/tests.
- Do not tune parameters unless explicitly requested.

Required output:

1. Goal and non-goals
2. Modules to inspect or change
3. Proposed implementation steps
4. Equations and conventions to preserve
5. Validation plan
6. Risks, assumptions, and open decisions
7. Recommended handoff order

## Gate 3: Implementation

Purpose: implement the approved plan.

Rules:

- Keep edits scoped.
- Do not change controller architecture unless requested.
- Do not alter state/input order or sign conventions.
- Keep code importable and tested.

Required output:

1. Files changed
2. Summary of edits
3. Equations or logic changed
4. Assumptions made
5. Risks introduced
6. Validation run

## Gate 4: Validation

Purpose: verify behavior without changing controller logic.

Rules:

- Do not tune the controller.
- Add tests, fixtures, or diagnostics only.
- Separate deterministic parity tests from solver-tolerance tests.

Required output:

1. Tests or fixtures created/updated
2. Variables checked
3. Expected pass/fail behavior
4. Numerical tolerances
5. Remaining uncertainties

## Gate 5: Experiment Summary

Purpose: interpret results and recommend next steps.

Required output:

1. What worked
2. What failed
3. Evidence from tests/logs
4. Likely cause
5. Recommended next experiment
6. Whether implementation should continue, pause, or roll back

