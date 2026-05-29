from dataclasses import dataclass

import numpy as np

from .config import TrailerLtvMpcConfig
from .math_utils import wrap_to_pi
from .models import trailer_model_continuous
from .reference_builder import ReferenceBundle


@dataclass(frozen=True)
class LtvModel:
    X_ref: np.ndarray
    Y_ref: np.ndarray
    U_ref: np.ndarray
    X_lin: np.ndarray
    U_lin: np.ndarray
    delta_T_ref: np.ndarray
    V2_ref: np.ndarray
    delta_T_lin: np.ndarray
    V2_lin: np.ndarray
    Ac: np.ndarray
    Bc: np.ndarray
    Ad: np.ndarray
    Bd: np.ndarray
    gd: np.ndarray
    Cy: np.ndarray | None
    dy: np.ndarray | None
    output_tracking_point: str

    @property
    def use_output_tracking(self) -> bool:
        return self.Cy is not None and self.dy is not None


def build_trailer_ltv_mpc_model(ref: ReferenceBundle, config: TrailerLtvMpcConfig) -> LtvModel:
    nx, nu, N = config.nx, config.nu, config.N
    Ac = np.zeros((nx, nx, N))
    Bc = np.zeros((nx, nu, N))
    Ad = np.zeros((nx, nx, N))
    Bd = np.zeros((nx, nu, N))
    gd = np.zeros((nx, N))
    Cy = np.zeros((nx, nx, N)) if ref.use_output_tracking else None
    dy = np.zeros((nx, N)) if ref.use_output_tracking else None

    for stage_idx in range(N):
        x_lin = ref.X_lin[:, stage_idx]
        psi2 = float(wrap_to_pi(x_lin[2]))
        delta_T = float(ref.delta_T_lin[stage_idx])
        V2 = float(ref.V2_lin[stage_idx])
        u_lin = np.array([delta_T, V2])

        Ac_k = np.array(
            [
                [0.0, 0.0, -V2 * np.sin(psi2)],
                [0.0, 0.0, V2 * np.cos(psi2)],
                [0.0, 0.0, 0.0],
            ]
        )
        sec_delta_sq = 1.0 / (np.cos(delta_T) ** 2)
        Bc_k = np.array(
            [
                [0.0, np.cos(psi2)],
                [0.0, np.sin(psi2)],
                [(V2 / config.geom.L2) * sec_delta_sq, np.tan(delta_T) / config.geom.L2],
            ]
        )
        Ad[:, :, stage_idx] = np.eye(nx) + config.Ts * Ac_k
        Bd[:, :, stage_idx] = config.Ts * Bc_k
        f_lin = trailer_model_continuous(x_lin, delta_T, V2, config.geom)
        gd[:, stage_idx] = config.Ts * (f_lin - Ac_k @ x_lin - Bc_k @ u_lin)
        Ac[:, :, stage_idx] = Ac_k
        Bc[:, :, stage_idx] = Bc_k

        if ref.use_output_tracking:
            Cy_k = np.array(
                [
                    [1.0, 0.0, -config.geom.L2 * np.sin(psi2)],
                    [0.0, 1.0, config.geom.L2 * np.cos(psi2)],
                    [0.0, 0.0, 1.0],
                ]
            )
            y_lin = np.array(
                [
                    x_lin[0] + config.geom.L2 * np.cos(psi2),
                    x_lin[1] + config.geom.L2 * np.sin(psi2),
                    psi2,
                ]
            )
            Cy[:, :, stage_idx] = Cy_k
            dy[:, stage_idx] = y_lin - Cy_k @ x_lin

    return LtvModel(
        X_ref=ref.X_ref,
        Y_ref=ref.Y_ref,
        U_ref=ref.U_ref,
        X_lin=ref.X_lin,
        U_lin=ref.U_lin,
        delta_T_ref=ref.delta_T_ref,
        V2_ref=ref.V2_ref,
        delta_T_lin=ref.delta_T_lin,
        V2_lin=ref.V2_lin,
        Ac=Ac,
        Bc=Bc,
        Ad=Ad,
        Bd=Bd,
        gd=gd,
        Cy=Cy,
        dy=dy,
        output_tracking_point=ref.output_tracking_point,
    )
