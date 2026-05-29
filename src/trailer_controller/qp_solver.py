from dataclasses import dataclass

import numpy as np

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - exercised on minimal Python installs
    sp = None

from .ltv_model import LtvModel


@dataclass(frozen=True)
class QpProblem:
    H: np.ndarray
    f: np.ndarray
    A: object
    lower: np.ndarray
    upper: np.ndarray
    rate_operator: np.ndarray
    rate_offset: np.ndarray
    Ax: np.ndarray
    Bu: np.ndarray
    D: np.ndarray


@dataclass(frozen=True)
class QpSolution:
    U: np.ndarray
    status: str
    objective: float
    iterations: int | None


class QpSolver:
    def solve(self, problem: QpProblem) -> QpSolution:
        if sp is None:
            raise RuntimeError("scipy is required for OSQP-based Trailer LTV MPC solving.")
        try:
            import osqp
        except ImportError as exc:
            raise RuntimeError("OSQP is required for Trailer LTV MPC solving.") from exc

        solver = osqp.OSQP()
        solver.setup(
            P=sp.csc_matrix(problem.H),
            q=problem.f,
            A=problem.A,
            l=problem.lower,
            u=problem.upper,
            verbose=False,
            polish=True,
        )
        result = solver.solve()
        if result.x is None or result.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP failed: {result.info.status}")
        objective = 0.5 * result.x @ problem.H @ result.x + problem.f @ result.x
        return QpSolution(
            U=np.asarray(result.x, dtype=float),
            status=str(result.info.status),
            objective=float(objective),
            iterations=int(result.info.iter),
        )


def build_qp_problem(
    model: LtvModel,
    Q,
    Qf,
    R,
    U0,
    X0,
    umin,
    umax,
    delta_umin,
    delta_umax,
    Rd,
) -> QpProblem:
    Ax, Bu, D = build_prediction_matrices(model)
    N = model.X_ref.shape[1]
    nu = model.Bd.shape[1]
    rate_operator, rate_offset, lb_u, ub_u, lbA, ubA = build_rate_constraints(
        N, nu, U0, umin, umax, delta_umin, delta_umax
    )
    H, f = build_cost_terms(Ax, Bu, D, Q, Qf, R, Rd, model, X0, rate_operator, rate_offset)
    if sp is None:
        A = np.vstack([np.eye(N * nu), rate_operator])
    else:
        A = sp.vstack([sp.eye(N * nu, format="csc"), sp.csc_matrix(rate_operator)], format="csc")
    lower = np.concatenate([lb_u, lbA])
    upper = np.concatenate([ub_u, ubA])
    return QpProblem(H, f, A, lower, upper, rate_operator, rate_offset, Ax, Bu, D)


def build_prediction_matrices(model: LtvModel):
    Ad, Bd, gd = model.Ad, model.Bd, model.gd
    nx = Ad.shape[0]
    nu = Bd.shape[1]
    N = Ad.shape[2]
    Ax = np.zeros((N * nx, nx))
    Bu = np.zeros((N * nx, N * nu))
    D = np.zeros(N * nx)
    Phi = np.eye(nx)
    Gamma = np.zeros((nx, N * nu))
    offset = np.zeros(nx)
    for stage_idx in range(N):
        A_k = Ad[:, :, stage_idx]
        B_k = Bd[:, :, stage_idx]
        g_k = gd[:, stage_idx]
        row = slice(stage_idx * nx, (stage_idx + 1) * nx)
        col = slice(stage_idx * nu, (stage_idx + 1) * nu)
        Phi = A_k @ Phi
        Gamma = A_k @ Gamma
        Gamma[:, col] = B_k
        offset = A_k @ offset + g_k
        Ax[row, :] = Phi
        Bu[row, :] = Gamma
        D[row] = offset
    return Ax, Bu, D


def build_cost_terms(Ax, Bu, D, Q, Qf, R, Rd, model: LtvModel, X0, rate_operator, rate_offset):
    tracking_Ax, tracking_Bu, tracking_D, tracking_ref = build_tracking_prediction_terms(Ax, Bu, D, model)
    ny, N = tracking_ref.shape
    Qbar = np.kron(np.eye(N), Q)
    terminal = slice((N - 1) * ny, N * ny)
    Qbar[terminal, terminal] = Qf
    nu = model.U_ref.shape[0]
    Rbar = _input_weight_matrix(R, N, nu)
    Rdbar = _input_weight_matrix(Rd, N, nu)
    tracking_ref_stack = tracking_ref.reshape(-1, order="F")
    U_ref_stack = model.U_ref.reshape(-1, order="F")
    H = 2.0 * (tracking_Bu.T @ Qbar @ tracking_Bu + Rbar + rate_operator.T @ Rdbar @ rate_operator)
    H = 0.5 * (H + H.T)
    f = 2.0 * (
        tracking_Bu.T @ Qbar @ (tracking_Ax @ X0 + tracking_D - tracking_ref_stack)
        - Rbar @ U_ref_stack
        - rate_operator.T @ Rdbar @ rate_offset
    )
    return H, f


def build_tracking_prediction_terms(Ax, Bu, D, model: LtvModel):
    if not model.use_output_tracking:
        return Ax, Bu, D, model.X_ref
    ny, nx, N = model.Cy.shape
    Cbar = np.zeros((N * ny, N * nx))
    dy_stack = np.zeros(N * ny)
    for stage_idx in range(N):
        row = slice(stage_idx * ny, (stage_idx + 1) * ny)
        col = slice(stage_idx * nx, (stage_idx + 1) * nx)
        Cbar[row, col] = model.Cy[:, :, stage_idx]
        dy_stack[row] = model.dy[:, stage_idx]
    return Cbar @ Ax, Cbar @ Bu, Cbar @ D + dy_stack, model.Y_ref


def build_rate_constraints(N, nu, U0, umin, umax, delta_umin, delta_umax):
    T = np.eye(N * nu) - np.diag(np.ones((N - 1) * nu), -nu)
    t0 = np.zeros(N * nu)
    t0[:nu] = np.asarray(U0, dtype=float).reshape(nu)
    lb_u = _expand_stagewise_bounds(umin, N, nu, "umin")
    ub_u = _expand_stagewise_bounds(umax, N, nu, "umax")
    lb_delta = _expand_stagewise_bounds(delta_umin, N, nu, "delta_umin")
    ub_delta = _expand_stagewise_bounds(delta_umax, N, nu, "delta_umax")
    return T, t0, lb_u, ub_u, lb_delta + t0, ub_delta + t0


def _expand_stagewise_bounds(bounds, N, nu, label):
    arr = np.asarray(bounds, dtype=float)
    if arr.size == nu:
        return np.tile(arr.reshape(nu), N)
    if arr.shape == (nu, N):
        return arr.reshape(-1, order="F")
    if arr.size == N * nu:
        return arr.reshape(N * nu)
    raise ValueError(f"{label} must be {nu}, {nu}x{N}, or {N * nu}.")


def _input_weight_matrix(weight, N, nu):
    arr = np.asarray(weight, dtype=float)
    if arr.size == 1:
        return float(arr) * np.eye(N * nu)
    return np.kron(np.eye(N), arr)
