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
    from drone_models._typing import Array  # To be changed to a Protocol later (see array-api#589)


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
    L: float,
    prop_inertia: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    rpm2thrust: Array,
    rpm2torque: Array,
    mixing_matrix: Array,
    drag_matrix: Array,
    rotor_dyn_coef: Array,
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
        L: Distance from the CoM to the motor (m).
        prop_inertia: Inertia of one propeller in z direction (kg m^2).
        gravity_vec: Gravity vector (m/s^2). We assume the gravity vector points downwards, e.g.
            [0, 0, -9.81].
        J: Inertia matrix (kg m^2).
        J_inv: Inverse inertia matrix (1/kg m^2).
        rpm2thrust: Propeller force constant (N min^2).
        rpm2torque: Propeller torque constant (Nm min^2).
        mixing_matrix: Mixing matrix denoting the turn direction of the motors (4x3).
        drag_matrix: Drag matrix containing the linear drag coefficients (3x3).
        rotor_dyn_coef: Rotor dynamics coefficients.

    .. math::
        \sum_{i=1}^{\\infty} x_{i} TODO

    Warning:
        Do not use quat_dot directly for integration! Only usage of ang_vel is mathematically correct.
        If you still decide to use quat_dot to integrate, ensure unit length!
        More information https://ahrs.readthedocs.io/en/latest/filters/angular.html
    """
    xp = array_namespace(pos)
    mass, L, prop_inertia, gravity_vec, rpm2thrust, rpm2torque = to_xp(
        mass, L, prop_inertia, gravity_vec, rpm2thrust, rpm2torque, xp=xp, device=device(pos)
    )
    J, J_inv, mixing_matrix, rotor_dyn_coef, drag_matrix = to_xp(
        J, J_inv, mixing_matrix, rotor_dyn_coef, drag_matrix, xp=xp, device=device(pos)
    )
    rot = R.from_quat(quat)  # from body to world
    rot_mat = rot.inv().as_matrix()  # from world to body
    # Rotor dynamics
    if rotor_vel is None:
        rotor_vel, rotor_vel_dot = cmd, None
    else:
        rotor_vel_dot = xp.where(
            cmd > rotor_vel,
            rotor_dyn_coef[0] * (cmd - rotor_vel) + rotor_dyn_coef[1] * (cmd**2 - rotor_vel**2),
            rotor_dyn_coef[2] * (cmd - rotor_vel) + rotor_dyn_coef[3] * (cmd**2 - rotor_vel**2),
        )
    # Creating force and torque vector
    forces_motor = rpm2thrust[0] + rpm2thrust[1] * rotor_vel + rpm2thrust[2] * rotor_vel**2
    forces_motor_tot = xp.sum(forces_motor, axis=-1)
    zeros = xp.zeros_like(forces_motor_tot)
    forces_motor_vec = xp.stack((zeros, zeros, forces_motor_tot), axis=-1)
    forces_motor_vec_world = rot.apply(forces_motor_vec)
    force_gravity = gravity_vec * mass
    force_drag = (rot_mat.mT @ (drag_matrix @ (rot_mat @ vel[..., None])))[..., 0]

    torques_motor = rpm2torque[0] + rpm2torque[1] * rotor_vel + rpm2torque[2] * rotor_vel**2
    torque_thrust = (mixing_matrix @ (forces_motor)[..., None])[..., 0] * xp.stack(
        [L, L, xp.asarray(0.0)]
    )
    torque_drag = (mixing_matrix @ (torques_motor)[..., None])[..., 0] * xp.stack(
        [xp.asarray(0.0), xp.asarray(0.0), xp.asarray(1.0)]
    )
    # convert rotor speed from RPM to rad/s for physical calculations
    rpm_to_rad = 2 * xp.pi / 60
    rotor_vel_rads = rotor_vel * rpm_to_rad
    rotor_vel_dot_rads = (
        rotor_vel_dot * rpm_to_rad if rotor_vel_dot is not None else xp.zeros_like(rotor_vel)
    )
    torque_inertia = prop_inertia * xp.stack(
        [
            -ang_vel[..., 1] * xp.sum(mixing_matrix[..., -1, :] * rotor_vel_rads, axis=-1),
            -ang_vel[..., 0] * xp.sum(mixing_matrix[..., -1, :] * rotor_vel_rads, axis=-1),
            xp.sum(mixing_matrix[..., -1, :] * rotor_vel_dot_rads, axis=-1),
        ],
        axis=-1,
    )
    torque_vec = torque_thrust + torque_drag + torque_inertia

    # Linear equation of motion
    forces_sum = forces_motor_vec_world + force_gravity + force_drag
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
    L: float,
    prop_inertia: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    rpm2thrust: Array,
    rpm2torque: Array,
    mixing_matrix: Array,
    rotor_dyn_coef: Array,
    drag_matrix: Array,
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
        # Rotor dynamics
        rotor_vel_dot = cs.if_else(
            U > symbols.rotor_vel,
            rotor_dyn_coef[0] * (U - symbols.rotor_vel)
            + rotor_dyn_coef[1] * (U**2 - symbols.rotor_vel**2),
            rotor_dyn_coef[2] * (U - symbols.rotor_vel)
            + rotor_dyn_coef[3] * (U**2 - symbols.rotor_vel**2),
        )
    else:
        symbols.rotor_vel = U
    # Creating force and torque vector
    forces_motor = (
        rpm2thrust[0] + rpm2thrust[1] * symbols.rotor_vel + rpm2thrust[2] * symbols.rotor_vel**2
    )
    forces_motor_vec = cs.vertcat(0.0, 0.0, cs.sum1(forces_motor))
    forces_motor_vec_world = symbols.rot @ forces_motor_vec
    force_gravity = gravity_vec * mass
    force_drag = symbols.rot @ (drag_matrix @ (symbols.rot.T @ symbols.vel))

    torques_motor = (
        rpm2torque[0] + rpm2torque[1] * symbols.rotor_vel + rpm2torque[2] * symbols.rotor_vel**2
    )
    torques_thrust = mixing_matrix @ forces_motor * cs.vertcat(L, L, 0.0)
    torques_drag = mixing_matrix @ torques_motor * cs.vertcat(0.0, 0.0, 1.0)
    # convert rotor speed from RPM to rad/s for physical calculations
    rpm_to_rad = 2 * cs.pi / 60
    rotor_vel_rads = symbols.rotor_vel * rpm_to_rad
    rotor_vel_dot_rads = rotor_vel_dot * rpm_to_rad if model_rotor_vel else symbols.rotor_vel * 0.0
    torque_inertia = prop_inertia * cs.vertcat(
        -symbols.ang_vel[1] * cs.sum(mixing_matrix[-1, :] * rotor_vel_rads),
        -symbols.ang_vel[0] * cs.sum(mixing_matrix[-1, :] * rotor_vel_rads),
        cs.sum(mixing_matrix[-1, :] * rotor_vel_dot_rads),
    )
    torques_motor_vec = torques_thrust + torques_drag + torque_inertia

    # Linear equation of motion
    forces_sum = forces_motor_vec_world + force_gravity + force_drag
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
