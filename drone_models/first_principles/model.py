"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
from array_api_compat import array_namespace, device
from scipy.spatial.transform import Rotation as R

import drone_models.symbols as symbols
from drone_models.core import supports
from drone_models.utils import rotation, to_xp

if TYPE_CHECKING:
    from array_api_typing import Array


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
    rpm2thrust: Array,
    rpm2torque: Array,
    L: float,
    mixing_matrix: Array,
    rotor_tau: float,
) -> tuple[Array, Array, Array, Array, Array | None]:
    r"""First principles model for a quatrotor.

    The input consists of four forces in [N]. TODO more detail.

    Based on the quaternion model from https://www.dynsyslab.org/wp-content/papercite-data/pdf/mckinnon-robot20.pdf

    Args:
        pos: Position of the drone (m).
        quat: Quaternion of the drone (xyzw).
        vel: Velocity of the drone (m/s).
        ang_vel: Angular velocity of the drone (rad/s).
        cmd: Motor speeds (RPMs).
        rotor_vel: Angular velocity of the 4 motors (RPMs). If None, the commanded thrust is
            directly applied. If value is given, thrust dynamics are calculated.
        dist_f: Disturbance force (N) in the world frame acting on the CoM.
        dist_t: Disturbance torque (Nm) in the world frame acting on the CoM.

        mass: Mass of the drone (kg).
        gravity_vec: Gravity vector (m/s^2). We assume the gravity vector points downwards, e.g.
            [0, 0, -9.81].
        J: Inertia matrix (kg m^2).
        J_inv: Inverse inertia matrix (1/kg m^2).
        rpm2thrust: Propeller force constant (N min^2).
        rpm2torque: Propeller torque constant (Nm min^2).
        L: Distance from the CoM to the motor (m).
        mixing_matrix: Mixing matrix denoting the turn direction of the motors (4x3).
        thrust_tau: Thrust time constant (s).

    .. math::
        \sum_{i=1}^{\\infty} x_{i} TODO

    Warning:
        Do not use quat_dot directly for integration! Only usage of ang_vel is mathematically correct.
        If you still decide to use quat_dot to integrate, ensure unit length!
        More information https://ahrs.readthedocs.io/en/latest/filters/angular.html
    """
    xp = array_namespace(pos)
    mass, gravity_vec, J, J_inv, rpm2thrust, rpm2torque, L, mixing_matrix, rotor_tau = to_xp(
        mass,
        gravity_vec,
        J,
        J_inv,
        rpm2thrust,
        rpm2torque,
        L,
        mixing_matrix,
        rotor_tau,
        xp=xp,
        device=device(pos),
    )
    rot = R.from_quat(quat)
    # Rotor dynamics
    if rotor_vel is None:
        rotor_vel, rotor_vel_dot = cmd, None
    else:
        rotor_vel_dot = 1 / rotor_tau * (cmd - rotor_vel)  # - 1 / KM * rotor_vel**2
    # Creating force and torque vector
    forces_motor = rpm2thrust[0] + rpm2thrust[1] * rotor_vel + rpm2thrust[2] * rotor_vel**2
    torques_motor = rpm2torque[0] + rpm2torque[1] * rotor_vel + rpm2torque[2] * rotor_vel**2
    forces_motor_tot = xp.sum(forces_motor, axis=-1)
    zeros = xp.zeros_like(forces_motor_tot)
    forces_motor_vec = xp.stack((zeros, zeros, forces_motor_tot), axis=-1)

    torque_vec = (mixing_matrix @ (forces_motor)[..., None])[..., 0] * xp.stack(
        [L, L, xp.asarray(0.0)]
    ) + (mixing_matrix @ (torques_motor)[..., None])[..., 0] * xp.stack(
        [xp.asarray(0.0), xp.asarray(0.0), xp.asarray(1.0)]
    )

    # Linear equation of motion
    forces_motor_vec_world = rot.apply(forces_motor_vec)
    forces_sum = forces_motor_vec_world + gravity_vec * mass
    if dist_f is not None:
        forces_sum = forces_sum + dist_f

    pos_dot = vel
    vel_dot = forces_sum / mass

    # Rotational equation of motion
    if dist_t is not None:
        torque_vec = torque_vec + rot.apply(dist_t, inverse=True)
    quat_dot = rotation.ang_vel2quat_dot(quat, ang_vel)
    torque_vec = torque_vec - xp.linalg.cross(ang_vel, (J @ ang_vel[..., None])[..., 0])
    ang_vel_dot = (J_inv @ torque_vec[..., None])[..., 0]
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
    rpm2thrust: Array,
    rpm2torque: Array,
    L: float,
    mixing_matrix: Array,
    rotor_tau: float,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """TODO take from numeric."""
    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.quat, symbols.vel, symbols.ang_vel)
    if model_rotor_vel:
        X = cs.vertcat(X, symbols.rotor_vel)
    if model_dist_f:
        X = cs.vertcat(X, symbols.dist_f)
    if model_dist_t:
        X = cs.vertcat(X, symbols.dist_t)
    U = symbols.cmd_rotor_vel

    # Defining the dynamics function
    if model_rotor_vel:
        # Thrust dynamics
        rotor_vel_dot = 1 / rotor_tau * (U - symbols.rotor_vel)  # - 1 / KM * symbols.rotor_vel**2
        forces_motor = (
            rpm2thrust[0] + rpm2thrust[1] * symbols.rotor_vel + rpm2thrust[2] * symbols.rotor_vel**2
        )
        torques_motor = (
            rpm2torque[0] + rpm2torque[1] * symbols.rotor_vel + rpm2torque[2] * symbols.rotor_vel**2
        )
    else:
        forces_motor = rpm2thrust[0] + rpm2thrust[1] * U + rpm2thrust[2] * U**2
        torques_motor = rpm2torque[0] + rpm2torque[1] * U + rpm2torque[2] * U**2

    # Creating force and torque vector
    forces_motor_vec = cs.vertcat(0.0, 0.0, cs.sum1(forces_motor))
    torques_motor_vec = mixing_matrix @ forces_motor * cs.vertcat(
        L, L, 0.0
    ) + mixing_matrix @ torques_motor * cs.vertcat(0.0, 0.0, 1.0)

    # Linear equation of motion
    forces_motor_vec_world = symbols.rot @ forces_motor_vec
    forces_sum = forces_motor_vec_world + gravity_vec * mass
    if model_dist_f:
        forces_sum = forces_sum + symbols.dist_f

    pos_dot = symbols.vel
    vel_dot = forces_sum / mass

    # Rotational equation of motion
    xi = cs.vertcat(
        cs.horzcat(0, -symbols.ang_vel.T), cs.horzcat(symbols.ang_vel, -cs.skew(symbols.ang_vel))
    )
    quat_dot = 0.5 * (xi @ symbols.quat)
    torques_sum = torques_motor_vec
    if model_dist_t:
        torques_sum = torques_sum + symbols.rot.T @ symbols.dist_t
    ang_vel_dot = J_inv @ (torques_sum - cs.cross(symbols.ang_vel, J @ symbols.ang_vel))

    if model_rotor_vel:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot)
    else:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(symbols.pos, symbols.quat)

    return X_dot, X, U, Y
