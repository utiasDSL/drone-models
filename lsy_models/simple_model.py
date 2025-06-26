import casadi as cs
import numpy as np
import scipy.linalg

from lsy_models.utils.constants import Constants
constants = Constants.from_config("cf2x_L250")

def export_quadrotor_ode_model():
    """Symbolic Quadrotor Model."""
    
    """Model setting"""
    # define basic variables in state and input vector
    px, py, pz = cs.MX.sym("px"), cs.MX.sym("py"), cs.MX.sym("pz")
    pos = cs.vertcat(px, py, pz)  # Position
    vx, vy, vz = cs.MX.sym("vx"), cs.MX.sym("vy"), cs.MX.sym("vz")
    vel = cs.vertcat(vx, vy, vz)  # Velocity
    roll, pitch, yaw = cs.MX.sym("roll"), cs.MX.sym("pitch"), cs.MX.sym("yaw")
    rpy = cs.vertcat(roll, pitch, yaw)  # Euler angles
    droll, dpitch, dyaw = cs.MX.sym("droll"), cs.MX.sym("dpitch"), cs.MX.sym("dyaw")
    rpy_dot = cs.vertcat(droll, dpitch, dyaw)  # Euler angles dot
    thrust = cs.MX.sym("thrust")

    r_cmd = cs.MX.sym("r_cmd")
    p_cmd = cs.MX.sym("p_cmd")
    y_cmd = cs.MX.sym("y_cmd")
    thrust_cmd = cs.MX.sym("thrust_cmd")

    # define state and input vector
    states = cs.vertcat(pos, rpy, vel, rpy_dot, thrust)
    inputs = cs.vertcat(r_cmd, p_cmd, y_cmd, thrust_cmd)

    # Define nonlinear system dynamics
    pos_dot = vel
    rpy_dot = cs.vertcat(droll, dpitch, dyaw)
    z_axis = cs.vertcat(
        cs.cos(roll) * cs.sin(pitch) * cs.cos(yaw) + cs.sin(roll) * cs.sin(yaw),
        cs.cos(roll) * cs.sin(pitch) * cs.sin(yaw) - cs.sin(roll) * cs.cos(yaw),
        cs.cos(roll) * cs.cos(pitch),
    )
    thrust_scaled = constants.DI_DD_ACC[0] * thrust
    vel_dot = (
        z_axis * thrust_scaled / constants.MASS
        + constants.GRAVITY_VEC
        + 1 / constants.MASS * constants.DI_DD_ACC[2] * vel
        + 1 / constants.MASS * constants.DI_DD_ACC[3] * vel * cs.fabs(vel)
    )
    rpy_rates_dot = cs.vertcat(
        constants.DI_DD_ROLL[0] * roll
        + constants.DI_DD_ROLL[1] * droll
        + constants.DI_DD_ROLL[2] * r_cmd,
        constants.DI_DD_PITCH[0] * pitch
        + constants.DI_DD_PITCH[1] * dpitch
        + constants.DI_DD_PITCH[2] * p_cmd,
        constants.DI_DD_YAW[0] * yaw
        + constants.DI_DD_YAW[1] * dyaw
        + constants.DI_DD_YAW[2] * y_cmd,
    )
    thrust_dot = 1 / constants.DI_DD_ACC[1] * (thrust_cmd - thrust)
    states_dot = cs.vertcat(pos_dot, rpy_dot, vel_dot, rpy_rates_dot, thrust_dot)

    x_dot_func = cs.Function(
        "quadrotor_dynamics",
        [states, inputs],
        [states_dot],
        ["x", "u"],
        ["xdot"],
    )

    return x_dot_func, states_dot, states, inputs


class DroneModel:
    def __init__(self):
        self.x_dot_func, state_dot, state, input = export_quadrotor_ode_model()
        self.n_states = self.x_dot_func.mx_in(0).shape[0]
        self.n_controls = self.x_dot_func.mx_in(1).shape[0]
        self.GRAVITY_ACC = constants.GRAVITY
        self.MASS = constants.MASS

        # Create functions for A and B
        x = state
        u = input
        x_dot = state_dot
        A = cs.jacobian(x_dot, x)
        B = cs.jacobian(x_dot, u)
        self.f_A = cs.Function("f_A", [x, u], [A])
        self.f_B = cs.Function("f_B", [x, u], [B])

    def linearize(self, x_eq, u_eq):
        A_lin = np.array(self.f_A(x_eq, u_eq))
        B_lin = np.array(self.f_B(x_eq, u_eq))
        # print(f"Linearized at equilibrium point: x_eq = {x_eq}, u_eq = {u_eq}")
        # print(f"Linearized A: \n{A_lin}\n, B: \n{B_lin}")
        # check if the system is controllable
        if is_controllable(A_lin, B_lin):
            print("System is controllable.")
        else:
            print("System is not controllable.")
            # raise ValueError("The system is not controllable at the equilibrium point.")

        return A_lin, B_lin

    def dynamics(self, x, u):
        return np.array(self.x_dot_func(x, u)).flatten()

def is_controllable(A, B):
    n = A.shape[0]
    # Build the controllability matrix
    controllability_matrix = B
    for i in range(1, n):
        controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i) @ B))
    return np.linalg.matrix_rank(controllability_matrix) == n

