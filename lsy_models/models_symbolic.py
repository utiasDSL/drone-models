"""TODO.

In large parts taken from
https://github.com/middleyuan/safe-control-gym/blob/d5f4f4f1cea112cab84031453b28c9ef65537ff0/safe_control_gym/envs/gym_pybullet_drones/quadrotor.py
"""

import casadi as cs

import lsy_models.utils.rotation as R
from lsy_models.utils.constants import Constants

# States
px, py, pz = cs.MX.sym("px"), cs.MX.sym("py"), cs.MX.sym("pz")
pos = cs.vertcat(px, py, pz)  # Position
vx, vy, vz = cs.MX.sym("vx"), cs.MX.sym("vy"), cs.MX.sym("vz")
vel = cs.vertcat(vx, vy, vz)  # Velocity
qw, qx, qy, qz = cs.MX.sym("qw"), cs.MX.sym("qx"), cs.MX.sym("qy"), cs.MX.sym("qz")
quat = cs.vertcat(qx, qy, qz, qw)  # Quaternions
rot = R.cs_quat2matrix(quat)  # Rotation matrix from body to world frame
p, q, r = cs.MX.sym("p"), cs.MX.sym("q"), cs.MX.sym("r")
ang_vel = cs.vertcat(p, q, r)  # Angular velocity
f1, f2, f3, f4 = cs.MX.sym("f1"), cs.MX.sym("f2"), cs.MX.sym("f3"), cs.MX.sym("f4")
forces_motor = cs.vertcat(f1, f2, f3, f4)  # Motor thrust
fx, fy, fz = cs.MX.sym("fx"), cs.MX.sym("fy"), cs.MX.sym("fz")
forces_dist = cs.vertcat(fx, fy, fz)  # Disturbance forces
tx, ty, tz = cs.MX.sym("tx"), cs.MX.sym("ty"), cs.MX.sym("tz")
torques_dist = cs.vertcat(tx, ty, tz)  # Disturbance torques

# Inputs
cmd_f1, cmd_f2, cmd_f3, cmd_f4 = (
    cs.MX.sym("cmd_f1"),
    cs.MX.sym("cmd_f2"),
    cs.MX.sym("cmd_f3"),
    cs.MX.sym("cmd_f4"),
)
cmd_force = cs.vertcat(cmd_f1, cmd_f2, cmd_f3, cmd_f4)
cmd_roll, cmd_pitch, cmd_yaw = (
    cs.MX.sym("cmd_roll"),
    cs.MX.sym("cmd_pitch"),
    cs.MX.sym("cmd_yaw"),
)
cmd_thrust = cs.MX.sym("cmd_thrust")
cmd_rpyt = cs.vertcat(cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust)


def first_principles(
    constants: Constants,
    calc_forces_motor: bool = True,
    calc_forces_dist: bool = False,
    calc_torques_dist: bool = False,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """TODO take from numeric."""
    # States and Inputs
    X = cs.vertcat(pos, quat, vel, ang_vel)
    if calc_forces_motor:
        X = cs.vertcat(X, forces_motor)
    if calc_forces_dist:
        X = cs.vertcat(X, forces_dist)
    if calc_torques_dist:
        X = cs.vertcat(X, torques_dist)
    U = cs.vertcat(cmd_f1, cmd_f2, cmd_f3, cmd_f4)

    # Defining the dynamics function
    if calc_forces_motor:
        # Thrust dynamics
        forces_motor_dot = constants.THRUST_TAU * (U - forces_motor)  # TODO 1/tau
        # Creating force and torque vector
        forces_motor_vec = cs.vertcat(0, 0, cs.sum1(forces_motor))
        torques_motor_vec = (
            constants.SIGN_MATRIX.T
            @ forces_motor
            * cs.vertcat(constants.L, constants.L, constants.KM / constants.KF)
        )
    else:
        # Creating force and torque vector
        forces_motor_vec = cs.vertcat(0, 0, cs.sum1(U))
        torques_motor_vec = (
            constants.SIGN_MATRIX.T
            @ U
            * cs.vertcat(constants.L, constants.L, constants.KM / constants.KF)
        )

    # Linear equation of motion
    pos_dot = vel
    vel_dot = rot @ forces_motor_vec / constants.MASS + constants.GRAVITY_VEC
    if calc_forces_dist:
        # Adding force disturbances to the state
        vel_dot = vel_dot + forces_dist / constants.MASS

    # Rotational equation of motion
    xi = cs.vertcat(cs.horzcat(0, -ang_vel.T), cs.horzcat(ang_vel, -cs.skew(ang_vel)))
    quat_dot = 0.5 * (xi @ quat)
    ang_vel_dot = constants.J_INV @ (
        torques_motor_vec - cs.cross(ang_vel, constants.J @ ang_vel)
    )
    if calc_torques_dist:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque
        torque = constants.J @ ang_vel_dot - cs.cross(ang_vel, constants.J @ ang_vel)
        # adding torque
        torque = torque + torques_dist
        # back to angular acceleration
        ang_vel_dot = constants.J_INV @ torque

    if calc_forces_motor:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot, forces_motor_dot)
    else:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(pos, quat)

    return X_dot, X, U, Y


def f_fitted_DI_rpyt(constants: Constants) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """TODO."""
    return f_fitted_DI_rpyt_core(constants, calc_forces_motor=False)


def f_fitted_DI_D_rpyt(constants: Constants) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """TODO."""
    return f_fitted_DI_rpyt_core(constants, calc_forces_motor=True)


def f_fitted_DI_rpyt_core(
    constants: Constants,
    calc_forces_motor: bool = False,
    calc_forces_dist: bool = False,
    calc_torques_dist: bool = False,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """The fitted double integrator (DI) model with optional motor delay (D).

    TODO.
    """
    # States and Inputs
    X = cs.vertcat(pos, quat, vel, ang_vel)
    if calc_forces_motor:
        X = cs.vertcat(X, forces_motor)
    if calc_forces_dist:
        X = cs.vertcat(X, forces_dist)
    if calc_torques_dist:
        X = cs.vertcat(X, torques_dist)
    U = cs.vertcat(cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust)

    # Defining the dynamics function
    if calc_forces_motor:
        # Note: Due to the structure of the integrator, we split the commanded thrust into
        # four equal parts and later apply the sum as total thrust again. Those four forces
        # are not the true forces of the motors, but the sum is the true total thrust.
        forces_motor_dot = 1 / constants.DI_D_ACC[2] * (cmd_thrust / 4 - forces_motor)
        thrust = cs.sum1(forces_motor)
        # Creating force vector
        forces_motor_vec = cs.vertcat(
            0, 0, constants.DI_D_ACC[0] + constants.DI_D_ACC[1] * thrust
        )
    else:
        thrust = cmd_thrust
        # Creating force vector
        forces_motor_vec = cs.vertcat(
            0, 0, constants.DI_ACC[0] + constants.DI_ACC[1] * thrust
        )

    # Linear equation of motion
    pos_dot = vel
    vel_dot = rot @ forces_motor_vec / constants.MASS + constants.GRAVITY_VEC
    if calc_forces_dist:
        # Adding force disturbances to the state
        vel_dot = vel_dot + forces_dist / constants.MASS

    # Rotational equation of motion
    euler_angles = R.cs_quat2euler(quat)

    xi = cs.vertcat(cs.horzcat(0, -ang_vel.T), cs.horzcat(ang_vel, -cs.skew(ang_vel)))
    quat_dot = 0.5 * (xi @ quat)
    rpy_rates = R.cs_ang_vel2rpy_rates(quat, ang_vel)
    if calc_forces_motor:
        rpy_rates_dot = (
            constants.DI_D_PARAMS[:, 0] * euler_angles
            + constants.DI_D_PARAMS[:, 1] * rpy_rates
            + constants.DI_D_PARAMS[:, 2] * cs.vertcat(cmd_roll, cmd_pitch, cmd_yaw)
        )
    else:
        rpy_rates_dot = (
            constants.DI_PARAMS[:, 0] * euler_angles
            + constants.DI_PARAMS[:, 1] * rpy_rates
            + constants.DI_PARAMS[:, 2] * cs.vertcat(cmd_roll, cmd_pitch, cmd_yaw)
        )
    ang_vel_dot = R.cs_rpy_rates_deriv2ang_vel_deriv(quat, rpy_rates, rpy_rates_dot)
    if calc_torques_dist:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque
        torque = constants.J @ ang_vel_dot - cs.cross(ang_vel, constants.J @ ang_vel)
        # adding torque
        torque = torque + torques_dist
        # back to angular acceleration
        ang_vel_dot = constants.J_INV @ torque

    if calc_forces_motor:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot, forces_motor_dot)
    else:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(pos, quat)

    return X_dot, X, U, Y
