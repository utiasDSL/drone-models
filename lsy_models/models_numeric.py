"""This file contains all the numeric models for a generic quatrotor drone. The parameters need to be stored in the corresponding xml file."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import lsy_models.utils.rotation as R

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor

    from lsy_models.utils.constants import Constants

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def quat_dot_from_ang_vel(quat: Array, ang_vel: Array) -> Array:
    """Calculates the quaternion derivative based on an angular velocity."""
    xp = quat.__array_namespace__()
    x, y, z = xp.split(ang_vel, 3, axis=-1)
    ang_vel_skew = xp.stack(
        [
            xp.concat((xp.zeros_like(x), -z, y), axis=-1),
            xp.concat((z, xp.zeros_like(x), -x), axis=-1),
            xp.concat((-y, x, xp.zeros_like(x)), axis=-1),
        ],
        axis=-2,
    )
    xi1 = xp.insert(-ang_vel, 0, 0, axis=-1)  # First line of xi
    xi2 = xp.concat((xp.expand_dims(ang_vel.T, axis=0).T, -ang_vel_skew), axis=-1)
    xi = xp.concat((xp.expand_dims(xi1, axis=-2), xi2), axis=-2)
    return 0.5 * xp.matvec(xi, quat)
    # return 0.5 * (xi @ quat[..., None]).squeeze(axis=-1)


def f_first_principles(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    command: Array,
    constants: Constants,
    forces_motor: Array | None = None,
    forces_dist: Array | None = None,
    torques_dist: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """First principles model for a quatrotor.

    The input consists of four forces in [N]. TODO more detail.

    Based on the quaternion model from https://www.dynsyslab.org/wp-content/papercite-data/pdf/mckinnon-robot20.pdf

    Warning:
        Do not use quat_dot directly for integration! Only usage of ang_vel is mathematically correct.
        If you still decide to use quat_dot to integrate, ensure unit length!
        More information https://ahrs.readthedocs.io/en/latest/filters/angular.html

    forces_motor are the four indiviual forces of the propellers in body frame
    forces_dist and torques_dist are vectors in world frame
    """
    xp = pos.__array_namespace__()  # This looks into the type of the position array and decides what implementation to use (numpy, jax, etc)
    rot = R.from_quat(quat)

    # Thrust dynamics
    if forces_motor is None:
        forces_motor_dot = None
        forces_motor = command
    else:
        forces_motor_dot = constants.THRUST_TAU * (command - forces_motor)  # TODO 1/TAU
    # Creating force and torque vector
    forces_motor_tot = xp.sum(forces_motor, axis=-1)
    # forces_motor_tot = xp.sum(
    #     command, axis=-1
    # )  # Without motor dynamics TODO make motor forces None
    zeros = xp.zeros_like(forces_motor_tot)
    forces_motor_vec = xp.stack((zeros, zeros, forces_motor_tot), axis=-1)
    # Torques in x & y are simply the force x distance.
    # Because there currently is no way to identify the z torque in relation to the thrust,
    # we rely on a old identified value that can compute rpm to torque.
    # force = kf * rpm², torque = km * rpm² => torque = km/kf*force
    torques_motor_vec = xp.vecmat(forces_motor, constants.SIGN_MATRIX) * xp.array(
        [constants.L, constants.L, constants.KM / constants.KF]
    )

    # Linear equation of motion
    forces_motor_vec_world = rot.apply(forces_motor_vec)

    force_world_frame = forces_motor_vec_world + constants.GRAVITY_VEC * constants.MASS
    if forces_dist is not None:
        force_world_frame = force_world_frame + forces_dist

    pos_dot = vel
    vel_dot = force_world_frame / constants.MASS

    # Rotational equation of motion

    torques = torques_motor_vec
    if torques_dist is not None:
        # paper: rot.as_matrix() @ torques_dist
        torques = torques + rot.apply(torques_dist)
    quat_dot = quat_dot_from_ang_vel(quat, ang_vel)
    ang_vel_dot = xp.matvec(
        constants.J_INV, torques - xp.cross(ang_vel, xp.matvec(constants.J, ang_vel))
    )

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, forces_motor_dot


def f_fitted_DI_rpyt(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    command: Array,
    constants: Constants,
    forces_motor: Array | None = None,
    forces_dist: Array | None = None,
    torques_dist: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """The fitted double integrator (DI) model.

    For full description see corresponding core function.
    """
    if forces_motor is not None:
        raise NotImplementedError("The fitted_DI_rpyt model does not support motor dynamics!")
    return f_fitted_DI_rpyt_core(
        pos, quat, vel, ang_vel, command, constants, forces_motor, forces_dist, torques_dist
    )


def f_fitted_DI_D_rpyt(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    command: Array,
    constants: Constants,
    forces_motor: Array | None = None,
    forces_dist: Array | None = None,
    torques_dist: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """The fitted double integrator with motor delay (DI_D) model.

    For full description see corresponding core function.
    """
    if forces_motor is None:
        logger.warning(
            "The fitted_DI_D_rpyt model only supports motor dynamics activated! Will continue without motor dynamics"
        )
    return f_fitted_DI_rpyt_core(
        pos, quat, vel, ang_vel, command, constants, forces_motor, forces_dist, torques_dist
    )


def f_fitted_DI_rpyt_core(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    command: Array,
    constants: Constants,
    forces_motor: Array | None = None,
    forces_dist: Array | None = None,
    torques_dist: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """The fitted double integrator (DI) model with optional motor delay (D).

    Args:
        pos (Array): Position of the drone (m)
        quat (Array): Quaternion of the drone (xyzw)
        vel (Array): Velocity of the drone (m/s)
        ang_vel (Array): Angular velocity of the drone (rad/s)
        command (Array): RPYT command (roll, pitch, yaw in rad, thrust in N)
        constants (Constants): Containing the constants of the drone
        forces_motor (Array | None, optional): Thrust of the 4 motors in N. Defaults to None.
            If None, the commanded thrust is directly applied. If value is given, thrust dynamics are calculated.
        forces_dist (Array | None, optional): _description_. Defaults to None.
        torques_dist (Array | None, optional): _description_. Defaults to None.

    Returns:
        tuple[Array, Array, Array, Array, Array | None]: _description_
    """
    xp = pos.__array_namespace__()
    # 13 states
    cmd_f = command[..., -1]
    cmd_rpy = command[..., 0:3]
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler("xyz")
    rpy_rates = R.ang_vel2rpy_rates(quat, ang_vel)

    if forces_motor is None:
        forces_motor_dot = None
        thrust = constants.DI_ACC[0] + constants.DI_ACC[1] * cmd_f
    else:
        # Note: Due to the structure of the integrator, we split the commanded thrust into
        # four equal parts and later apply the sum as total thrust again. Those four forces
        # are not the true forces of the motors, but the sum is the true total thrust.
        forces_motor_dot = 1 / constants.DI_D_ACC[2] * (cmd_f[..., None] / 4 - forces_motor)
        forces_sum = xp.sum(forces_motor, axis=-1)
        thrust = constants.DI_D_ACC[0] + constants.DI_D_ACC[1] * forces_sum

    drone_z_axis = rot.inv().as_matrix()[..., -1, :]

    pos_dot = vel
    vel_dot = 1.0 / constants.MASS * thrust[..., None] * drone_z_axis + constants.GRAVITY_VEC
    if forces_dist is not None:
        # Adding force disturbances to the state
        vel_dot = vel_dot + forces_dist / constants.MASS

    # Rotational equation of motion
    quat_dot = quat_dot_from_ang_vel(quat, ang_vel)
    if forces_motor is None:
        rpy_rates_dot = (
            constants.DI_PARAMS[:, 0] * euler_angles
            + constants.DI_PARAMS[:, 1] * rpy_rates
            + constants.DI_PARAMS[:, 2] * cmd_rpy
        )
    else:
        rpy_rates_dot = (
            constants.DI_D_PARAMS[:, 0] * euler_angles
            + constants.DI_D_PARAMS[:, 1] * rpy_rates
            + constants.DI_D_PARAMS[:, 2] * cmd_rpy
        )
    ang_vel_dot = R.rpy_rates_deriv2ang_vel_deriv(quat, rpy_rates, rpy_rates_dot)
    if torques_dist is not None:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque
        torque = xp.matvec(constants.J, ang_vel_dot) - xp.cross(
            ang_vel, xp.matvec(constants.J, ang_vel)
        )
        # adding torque
        torque = torque + torques_dist
        # back to angular acceleration
        ang_vel_dot = xp.matvec(constants.J_INV, torque)

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, forces_motor_dot


def f_fitted_DI_DD_rpyt(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    command: Array,
    constants: Constants,
    forces_motor: Array | None = None,
    forces_dist: Array | None = None,
    torques_dist: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """The fitted double integrator (DI) model with optional motor delay (D).

    Args:
        pos (Array): Position of the drone (m)
        quat (Array): Quaternion of the drone (xyzw)
        vel (Array): Velocity of the drone (m/s)
        ang_vel (Array): Angular velocity of the drone (rad/s)
        command (Array): RPYT command (roll, pitch, yaw in rad, thrust in N)
        constants (Constants): Containing the constants of the drone
        forces_motor (Array | None, optional): Thrust of the 4 motors in N. Defaults to None.
            If None, the commanded thrust is directly applied. If value is given, thrust dynamics are calculated.
        forces_dist (Array | None, optional): _description_. Defaults to None.
        torques_dist (Array | None, optional): _description_. Defaults to None.

    Returns:
        tuple[Array, Array, Array, Array, Array | None]: _description_
    """
    xp = pos.__array_namespace__()
    # 13 states
    cmd_f = command[..., -1]
    cmd_rpy = command[..., 0:3]
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler("xyz")
    rpy_rates = R.ang_vel2rpy_rates(quat, ang_vel)

    if forces_motor is None:
        raise NotImplementedError
    else:
        # Note: Due to the structure of the integrator, we split the commanded thrust into
        # four equal parts and later apply the sum as total thrust again. Those four forces
        # are not the true forces of the motors, but the sum is the true total thrust.
        forces_motor_dot = 1 / constants.DI_DD_ACC[1] * (cmd_f[..., None] / 4 - forces_motor)
        forces_sum = xp.sum(forces_motor, axis=-1)
        thrust = constants.DI_DD_ACC[0] * forces_sum

    drone_z_axis = rot.inv().as_matrix()[..., -1, :]

    pos_dot = vel
    vel_dot = (
        1 / constants.MASS * thrust[..., None] * drone_z_axis
        + constants.GRAVITY_VEC
        + 1 / constants.MASS * constants.DI_DD_ACC[2] * vel
        + 1 / constants.MASS * constants.DI_DD_ACC[3] * vel * xp.abs(vel)
    )
    if forces_dist is not None:
        vel_dot = vel_dot + forces_dist / constants.MASS

    # Rotational equation of motion
    quat_dot = quat_dot_from_ang_vel(quat, ang_vel)
    rpy_rates_dot = (
        constants.DI_DD_PARAMS[:, 0] * euler_angles
        + constants.DI_DD_PARAMS[:, 1] * rpy_rates
        + constants.DI_DD_PARAMS[:, 2] * cmd_rpy
    )
    ang_vel_dot = R.rpy_rates2ang_vel(quat, rpy_rates_dot)
    if torques_dist is not None:
        # adding disturbances to the state
        # adding torque is a little more complex:
        # angular acceleration can be converted to torque
        torque = xp.matvec(constants.J, ang_vel_dot) - xp.cross(
            ang_vel, xp.matvec(constants.J, ang_vel)
        )
        # adding torque
        torque = torque + torques_dist
        # back to angular acceleration
        ang_vel_dot = xp.matvec(constants.J_INV, torque)

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, forces_motor_dot
