import numpy as np

from demo_planner import straight_path
from trailer_ltv_mpc import TrailerLtvMpcConfig, TrailerLtvMpcController


def main():
    config = TrailerLtvMpcConfig()
    controller = TrailerLtvMpcController(config)
    path = straight_path(direction="reverse")
    plant_state = np.array([9.739, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_prev = np.array([0.0, -0.2])
    output = controller.step(plant_state, u_prev, -1.0, path, 0)
    print("delta_f:", output.command.delta_f)
    print("V1:", output.command.V1)
    print("delta_T:", output.command.delta_T)
    print("V2:", output.command.V2)
    print("solver:", output.debug["solver"].status)


if __name__ == "__main__":
    main()
