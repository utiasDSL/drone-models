"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
from array_api_compat import array_namespace
from array_api_compat import device as xp_device
from scipy.spatial.transform import Rotation as R

import drone_models.symbols as symbols
from drone_models.core import supports
from drone_models.utils import rotation, to_xp

if TYPE_CHECKING:
    from array_api_typing import Array

# Additional symbols specific to this model
roll, pitch, yaw = cs.MX.sym("roll"), cs.MX.sym("pitch"), cs.MX.sym("yaw")
rpy = cs.vertcat(roll, pitch, yaw)  # Euler angles
droll, dpitch, dyaw = cs.MX.sym("droll"), cs.MX.sym("dpitch"), cs.MX.sym("dyaw")
drpy = cs.vertcat(droll, dpitch, dyaw)  # Euler angle rates
ddroll, ddpitch, ddyaw = cs.MX.sym("ddroll"), cs.MX.sym("ddpitch"), cs.MX.sym("ddyaw")
rpy_ddot = cs.vertcat(ddroll, ddpitch, ddyaw)  # Euler angle rates derivatives


@supports(rotor_dynamics=True)
def dynamics(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    cmd: Array,
    rotor_vel: Array | None = None,
    dist_f: Array | None = None,
    dist_t: Array | None = None,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    thrust_time_coef: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
    drag_linear_coef: Array,
    drag_square_coef: Array,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """Fitted model with linear, second order rpy dynamics with thrust dynamics and drag.

    Args:
        pos: Position of the drone (m).
        quat: Quaternion of the drone (xyzw).
        vel: Velocity of the drone (m/s).
        ang_vel: Angular velocity of the drone (rad/s).
        cmd: Roll pitch yaw (rad) and collective thrust (N) command.
        rotor_vel: Speed of the 4 motors (RPMs). If None, the commanded thrust is directly
            applied (not recommended). If value is given, rotor dynamics are calculated.
        dist_f: Disturbance force (N) in the world frame acting on the CoM.
        dist_t: Disturbance torque (Nm) in the world frame acting on the CoM.

        mass: Mass of the drone (kg).
        gravity_vec: Gravity vector (m/s^2). We assume the gravity vector points downwards, e.g.
            [0, 0, -9.81].
        J: Inertia matrix (kg m^2).
        J_inv: Inverse inertia matrix (1/kg m^2).
        thrust_time_coef: Coefficient for the rotor dynamics (1/s).
        acc_coef: Coefficient for the acceleration (1/s^2).
        cmd_f_coef: Coefficient for the collective thrust (N/rad^2).
        rpy_coef: Coefficient for the roll pitch yaw dynamics (1/s).
        rpy_rates_coef: Coefficient for the roll pitch yaw rates dynamics (1/s^2).
        cmd_rpy_coef: Coefficient for the roll pitch yaw command dynamics (1/s).
        drag_linear_coef: Coefficient for the linear drag (1/s).
        drag_square_coef: Coefficient for the square drag (1/s).

    Returns:
        The derivatives of all state variables.
    """
    xp = array_namespace(pos)
    device = xp_device(pos)
    # Convert constants to the correct framework and device
    mass, gravity_vec, J, J_inv = to_xp(mass, gravity_vec, J, J_inv, xp=xp, device=device)
    thrust_time_coef, acc_coef, cmd_f_coef = to_xp(
        thrust_time_coef, acc_coef, cmd_f_coef, xp=xp, device=device
    )
    rpy_coef, rpy_rates_coef, cmd_rpy_coef = to_xp(
        rpy_coef, rpy_rates_coef, cmd_rpy_coef, xp=xp, device=device
    )
    drag_linear_coef, drag_square_coef = to_xp(
        drag_linear_coef, drag_square_coef, xp=xp, device=device
    )
    cmd_f = cmd[..., -1]
    cmd_rpy = cmd[..., 0:3]
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler("xyz")

    # Note that we are abusing the rotor_vel state as the thrust
    if rotor_vel is None:
        rotor_vel, rotor_vel_dot = cmd_f[..., None], None
    else:
        rotor_vel_dot = 1 / thrust_time_coef * (cmd_f[..., None] - rotor_vel)

    forces_motor = rotor_vel[..., 0]
    thrust = acc_coef + cmd_f_coef * forces_motor

    drone_z_axis = rot.inv().as_matrix()[..., -1, :]

    pos_dot = vel
    vel_dot = (
        1 / mass * thrust[..., None] * drone_z_axis
        + gravity_vec
        + 1 / mass * drag_linear_coef * vel
        + 1 / mass * drag_square_coef * vel * xp.abs(vel)
    )
    if dist_f is not None:
        vel_dot = vel_dot + dist_f / mass

    # Rotational equation of motion
    quat_dot = rotation.ang_vel2quat_dot(quat, ang_vel)
    rpy_rates = rotation.ang_vel2rpy_rates(quat, ang_vel)
    rpy_rates_dot = rpy_coef * euler_angles + rpy_rates_coef * rpy_rates + cmd_rpy_coef * cmd_rpy
    ang_vel_dot = rotation.rpy_rates_deriv2ang_vel_deriv(quat, rpy_rates, rpy_rates_dot)
    if dist_t is not None:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque given the inertia matrix
        torque = (J @ ang_vel_dot[..., None])[..., 0]
        torque = torque + xp.linalg.cross(ang_vel, (J @ ang_vel[..., None])[..., 0])
        # adding torque. TODO: This should be a linear transformation. Can't we just transform the
        # disturbance torque to an ang_vel_dot summand directly?
        torque = torque + rot.apply(dist_t, inverse=True)
        # back to angular acceleration
        torque = torque - xp.linalg.cross(ang_vel, (J @ ang_vel[..., None])[..., 0])
        ang_vel_dot = (J_inv @ torque[..., None])[..., 0]

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot


def symbolic_dynamics(
    model_rotor_vel: bool = True,
    model_dist_f: bool = False,
    model_dist_t: bool = False,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    thrust_time_coef: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
    drag_linear_coef: Array,
    drag_square_coef: Array,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """Fitted model with linear, second order rpy dynamics with thrust dynamics and drag.

    For info on the args, see above.

    This wrapper converts the actual symbolic model, defined below, into the quat and ang_vel form.
    """
    # We need to set the rpy and drpy symbols before building the euler model
    symbols.rpy = rotation.cs_quat2euler(symbols.quat)
    symbols.drpy = rotation.cs_ang_vel2rpy_rates(symbols.quat, symbols.ang_vel)
    X_dot_euler, X_euler, U_euler, Y_euler = symbolic_dynamics_euler(
        model_rotor_vel=model_rotor_vel,
        mass=mass,
        gravity_vec=gravity_vec,
        J=J,
        J_inv=J_inv,
        thrust_time_coef=thrust_time_coef,
        acc_coef=acc_coef,
        cmd_f_coef=cmd_f_coef,
        rpy_coef=rpy_coef,
        rpy_rates_coef=rpy_rates_coef,
        cmd_rpy_coef=cmd_rpy_coef,
        drag_linear_coef=drag_linear_coef,
        drag_square_coef=drag_square_coef,
    )

    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.quat, symbols.vel, symbols.ang_vel)
    if model_rotor_vel:
        X = cs.vertcat(X, symbols.rotor_vel)
    if model_dist_f:
        X = cs.vertcat(X, symbols.dist_f)
    if model_dist_t:
        X = cs.vertcat(X, symbols.dist_t)
    U = U_euler

    # Linear equation of motion
    pos_dot = X_dot_euler[0:3]
    vel_dot = X_dot_euler[6:9]
    if model_dist_f:
        # Adding force disturbances to the state
        vel_dot = vel_dot + symbols.dist_f / mass

    # Rotational equation of motion
    xi = cs.vertcat(
        cs.horzcat(0, -symbols.ang_vel.T), cs.horzcat(symbols.ang_vel, -cs.skew(symbols.ang_vel))
    )
    quat_dot = 0.5 * (xi @ symbols.quat)
    ang_vel_dot = rotation.cs_rpy_rates_deriv2ang_vel_deriv(
        symbols.quat, symbols.drpy, X_dot_euler[9:12]
    )
    if model_dist_t:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque
        torque = J @ ang_vel_dot + cs.cross(symbols.ang_vel, J @ symbols.ang_vel)
        # adding torque
        torque = torque + symbols.rot.T @ symbols.dist_t
        # back to angular acceleration
        ang_vel_dot = J_inv @ (torque - cs.cross(symbols.ang_vel, J @ symbols.ang_vel))

    if model_rotor_vel:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot, X_dot_euler[-4:])
    else:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(symbols.pos, symbols.quat)

    return X_dot, X, U, Y


def symbolic_dynamics_euler(
    model_rotor_vel: bool = True,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    thrust_time_coef: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
    drag_linear_coef: Array,
    drag_square_coef: Array,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """The fitted linear, second order rpy dynamics with thrust dynamics and drag.

    For info on the args, see above.

    This function returns the actual model, as defined in the paper, for direct use.
    """
    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.rpy, symbols.vel, symbols.drpy)
    if model_rotor_vel:
        X = cs.vertcat(X, symbols.rotor_vel)
    U = symbols.cmd_rpyt
    cmd_rpy = U[:3]
    cmd_thrust = U[-1]
    rot = rotation.cs_rpy2matrix(symbols.rpy)

    # Defining the dynamics function
    # Note that we are abusing the rotor_vel state as the thrust
    if model_rotor_vel:
        rotor_vel_dot = 1 / thrust_time_coef * (cmd_thrust - symbols.rotor_vel)
        forces_motor = symbols.rotor_vel[0]  # We are only using the first element
    else:
        forces_motor = cmd_thrust

    # Creating force vector
    forces_motor_vec = cs.vertcat(0, 0, acc_coef + cmd_f_coef * forces_motor)

    # Linear equation of motion
    pos_dot = symbols.vel
    vel_dot = (
        rot @ forces_motor_vec / mass
        + gravity_vec
        + 1 / mass * drag_linear_coef * symbols.vel
        + 1 / mass * drag_square_coef * symbols.vel * cs.fabs(symbols.vel)
    )

    ddrpy = rpy_coef * symbols.rpy + rpy_rates_coef * symbols.drpy + cmd_rpy_coef * cmd_rpy

    if model_rotor_vel:
        X_dot = cs.vertcat(pos_dot, symbols.drpy, vel_dot, ddrpy, rotor_vel_dot)
    else:
        X_dot = cs.vertcat(pos_dot, symbols.drpy, vel_dot, ddrpy)
    Y = cs.vertcat(symbols.pos, symbols.rpy)

    return X_dot, X, U, Y
