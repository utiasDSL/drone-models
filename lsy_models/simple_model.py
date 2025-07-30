import sys
import casadi as cs
import numpy as np
import scipy.linalg
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

from lsy_models.utils.constants import Constants
from lsy_models.models_symbolic import three_d_attitude_delay
from lsy_models.models_numeric import f_three_d_attitude_delay
constants = Constants.from_config("cf2x_L350")
thrust_dynamics = True  # Use thrust dynamics
no_yaw = False  # Use yaw dynamics

def export_quadrotor_ode_model():
    """Symbolic Quadrotor Model using THREE_D_ATTITUDE_DELAY."""
    
    # Use the THREE_D_ATTITUDE_DELAY model from models_symbolic
    X_dot, X, U, Y = three_d_attitude_delay(
        constants=constants,
        calc_forces_motor=True,  # Include motor dynamics (13 states)
        calc_forces_dist=False,
        calc_torques_dist=False
    )
    
    # Create CasADi function
    x_dot_func = cs.Function(
        "quadrotor_dynamics",
        [X, U],
        [X_dot],
        ["x", "u"],
        ["xdot"],
    )
    
    return x_dot_func, X_dot, X, U


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
        # print(f"x_dot shape: {x_dot.shape}, x shape: {x.shape}, u shape: {u.shape}")
        A = cs.jacobian(x_dot, x)
        B = cs.jacobian(x_dot, u)
        # print(f"A shape: {A.shape}, B shape: {B.shape}")
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

        return A_lin, B_lin

    def dynamics(self, x, u):
        return np.array(self.x_dot_func(x, u)).flatten()

def is_controllable(A, B):
    n = A.shape[0]
    # Build the controllability matrix
    controllability_matrix = B
    for i in range(1, n):
        controllability_matrix = np.hstack(
            (controllability_matrix, np.linalg.matrix_power(A, i) @ B)
        )
    rankC = np.linalg.matrix_rank(controllability_matrix)
    print(f"Controllability matrix rank: {rankC}, expected rank: {n}")
    if rankC < n:
        # Decompose to find uncontrollable subspace basis
        U, S, Vt = np.linalg.svd(controllability_matrix, full_matrices=False)
        M_u = U[:, rankC:]  # Columns forming the uncontrollable subspace
        A_u = M_u.T @ A @ M_u

        # print the index of uncontrollable directions
        print("Uncontrollable directions (basis vectors):")
        for i in range(M_u.shape[1]):
            print(f"v{i+1}: {M_u[:, i]}")

        # Check stability of this uncontrollable subsystem
        eigvals_u, _ = np.linalg.eig(A_u)
        stable_unctr = np.all(np.abs(eigvals_u) < 1)
        if stable_unctr:
            print("Uncontrollable subsystem is stable => system is stablizable.")
        else:
            print("Uncontrollable subsystem is not stable => system is not stablizable.")
    else:
        print("System is fully controllable => stablizable.")

    return rankC == n

