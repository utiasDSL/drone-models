import numpy as np

def generate_figure_eight_trajectory(t_vec, drone, traj_period=25.0, scaling=1., z_height=1.0):
    """Generate a figure-eight trajectory for the 13-state drone model."""
    nx = drone.n_states
    
    traj_freq = 2.0 * np.pi / traj_period
    
    x_ref = scaling * np.sin(traj_freq * t_vec)
    y_ref = scaling * np.sin(traj_freq * t_vec) * np.cos(traj_freq * t_vec)
    z_ref = np.full_like(t_vec, z_height)
    
    x_dot_ref = scaling * traj_freq * np.cos(traj_freq * t_vec)
    y_dot_ref = scaling * traj_freq * (np.cos(traj_freq * t_vec)**2 - np.sin(traj_freq * t_vec)**2)
    z_dot_ref = np.zeros_like(t_vec)
    
    # Reference for angular states is zero, including yaw
    phi_ref, theta_ref, psi_ref, phi_dot_ref, theta_dot_ref, psi_dot_ref = [np.zeros_like(t_vec) for _ in range(6)]
    
    # Equilibrium motor force for hovering
    forces_motor_ref = np.full_like(t_vec, drone.GRAVITY_ACC * drone.MASS)

    if nx == 13:
        X_ref = np.vstack([
            x_ref, y_ref, z_ref,
            phi_ref, theta_ref, psi_ref,
            x_dot_ref, y_dot_ref, z_dot_ref,
            phi_dot_ref, theta_dot_ref, psi_dot_ref,
            forces_motor_ref
        ])
    elif nx == 12:
        X_ref = np.vstack([
            x_ref, y_ref, z_ref,
            phi_ref, theta_ref, psi_ref,
            x_dot_ref, y_dot_ref, z_dot_ref,
            phi_dot_ref, theta_dot_ref, psi_dot_ref,
        ])
    elif nx == 10:
        X_ref = np.vstack([
            x_ref, y_ref, z_ref,
            phi_ref, theta_ref,
            x_dot_ref, y_dot_ref, z_dot_ref,
            phi_dot_ref, theta_dot_ref
        ])

    return X_ref