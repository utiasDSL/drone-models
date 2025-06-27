import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from lsy_models.simple_model import DroneModel, thrust_dynamics, no_yaw
from lsy_models.simple_figure_eight import generate_figure_eight_trajectory
from lsy_models.lqr import design_lqr_controller


def main():
    """Main function to run LQR controller design and simulation."""
    # 1. System model
    drone = DroneModel()

    # 2. Linearize around hover
    z_height = 1.0
    x_eq = np.zeros(drone.n_states)
    
    forces_motor_eq = drone.GRAVITY_ACC * drone.MASS
    if thrust_dynamics:
        x_eq[12] = forces_motor_eq

    u_eq = np.zeros(drone.n_controls)
    u_eq[-1] = forces_motor_eq
    print(f"X_EQ: {x_eq}, U_EQ: {u_eq}")

    A, B = drone.linearize(x_eq, u_eq)

    # 3. LQR Controller Design
    T_sim = 30
    period = 15
    dt = 1/60

    # State penalty matrix
    if thrust_dynamics:
        Q = np.diag([
            10.0, 10.0, 20.0, # x, y , z    
            1.0, 1.0, 2.0, # roll, pitch, yaw
            1.0, 1.0, 1.0, # x_dot, y_dot, z_dot
            0.5, 0.5, 0.5, # roll_dot, pitch_dot, yaw_dot
            1.0             
        ])
    else:
        Q = np.diag([
            10.0, 10.0, 20.0, # x, y , z    
            1.0, 1.0, 2.0, # roll, pitch, yaw
            1.0, 1.0, 1.0, # x_dot, y_dot, z_dot
            0.5, 0.5, 0.5, # roll_dot, pitch_dot, yaw_dot
        ])
    if no_yaw:
        Q = np.diag([
            10.0, 10.0, 20.0, # x, y , z    
            1.0, 1.0, # roll, pitch,
            1.0, 1.0, 1.0, # x_dot, y_dot, z_dot
            0.5, 0.5, # roll_dot, pitch_dot
        ])
    # Control penalty matrix
    R = np.diag([0.5, 0.5, 0.5, 0.5])  # RPYT
    if no_yaw:
        R = np.diag([0.5, 0.5, 0.5])

    # Create LQR controller
    lqr_controller = design_lqr_controller(A, B, Q, R, dt)
    # print(f"LQR gain matrix K:\n{lqr_controller.get_gain_matrix()}")

    # 4. Reference Trajectory
    t_vec = np.arange(0, T_sim, dt)
    X_ref = generate_figure_eight_trajectory(t_vec, 
                                             drone, 
                                             traj_period=period,
                                             z_height=z_height)

    # 5. Simulation
    # Initial state
    x0 = np.zeros(drone.n_states)
    x0[2] = z_height # Start at the beginning of the trajectory's z height
    if thrust_dynamics:
        x0[12] = forces_motor_eq

    # Arrays to store simulation data
    X_sim = np.zeros((drone.n_states, len(t_vec)))
    U_sim = np.zeros((drone.n_controls, len(t_vec)))
    X_sim[:, 0] = x0

    # Simulation loop
    for i in range(len(t_vec) - 1):
        # Current state and reference
        x_current = X_sim[:, i]
        x_ref = X_ref[:, i]

        # LQR control law
        u_current = lqr_controller.compute_control(x_current, x_ref, u_eq)
        
        # clip the last action between 0.08 and 0.45 N
        u_current = np.clip(u_current, -0.5, 0.5)  # Clip roll, pitch, yaw commands
        u_current[-1] = np.clip(u_current[-1], 0.08, 0.45) 

        # Log control input
        U_sim[:, i] = u_current

        # if not no_yaw:
        #     print(f"yaw_action: {u_current[2]}, ")

        # Get dynamics from non-linear model
        x_dot = drone.dynamics(x_current, u_current)
        
        # Integrate one step forward (Euler integration)
        X_sim[:, i+1] = x_current + dt * x_dot

    # Log the last control input
    U_sim[:, -1] = lqr_controller.compute_control(X_sim[:, -1], X_ref[:, -1], u_eq)

    # 6. Plotting results
    _plot_results(t_vec, X_sim, X_ref, U_sim, drone)


def _plot_results(t_vec, X_sim, X_ref, U_sim, drone):
    """Generate all plotting results."""
    # 3D trajectory plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_sim[0, :], X_sim[1, :], X_sim[2, :], label='Actual Trajectory')
    ax.plot(X_ref[0, :], X_ref[1, :], X_ref[2, :], 'r--', label='Reference Trajectory')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Figure-Eight Trajectory Tracking')
    ax.legend()
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig('lqr_figure_eight_trajectory.png')

    # States vs time plot
    fig, axs = plt.subplots(7, 2, figsize=(15, 18), sharex=True)
    fig.suptitle('State Trajectories vs. Time')
    state_labels = [
        'x (m)', 'y (m)', 'z (m)',
        'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)',
        'x_dot (m/s)', 'y_dot (m/s)', 'z_dot (m/s)',
        'Roll_dot (rad/s)', 'Pitch_dot (rad/s)', 'Yaw_dot (rad/s)',
        'Motor Force (N)' 
    ]
    if no_yaw:
        state_labels = [
            'x (m)', 'y (m)', 'z (m)',
            'Roll (rad)', 'Pitch (rad)',
            'x_dot (m/s)', 'y_dot (m/s)', 'z_dot (m/s)',
            'Roll_dot (rad/s)', 'Pitch_dot (rad/s)'
        ]
    for i in range(drone.n_states):
        row, col = divmod(i, 2)
        axs[row, col].plot(t_vec, X_sim[i, :], label='Actual')
        if i < X_ref.shape[0]:
            axs[row, col].plot(t_vec, X_ref[i, :], 'r--', label='Reference')
        axs[row, col].set_ylabel(state_labels[i])
        axs[row, col].grid(True)
        axs[row, col].legend()
    
    # Hide the unused subplot
    if drone.n_states % 2 != 0:
        axs[-1, -1].set_visible(False)

    axs[-1, 0].set_xlabel('Time (s)')
    if drone.n_states > 1:
        axs[-1, 1].set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('lqr_figure_eight_results.png')

    # plot action vs time
    fig, axs = plt.subplots(drone.n_controls, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Control Inputs vs. Time')
    control_labels = ['Roll Cmd (rad)', 'Pitch Cmd (rad)', 'Yaw Cmd (rad)', 'Thrust Cmd (Newton)']
    if no_yaw:
        control_labels = ['Roll Cmd (rad)', 'Pitch Cmd (rad)', 'Thrust Cmd (Newton)']
    for i in range(drone.n_controls):
        axs[i].plot(t_vec, U_sim[i, :], label=control_labels[i])
        axs[i].set_ylabel(control_labels[i])
        axs[i].grid(True)
        axs[i].legend()
    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('lqr_figure_eight_inputs.png')
    # plt.show()


if __name__ == '__main__':
    main()

