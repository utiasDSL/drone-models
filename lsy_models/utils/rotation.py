"""In this file, some important rotations from scipy.spatial.transform.rotation are reimplemented in the Array API to be useable with jax, numpy, etc.

https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/spatial/transform.py
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import casadi as cs
import jax.numpy as jp
from jax.scipy.spatial.transform import Rotation as JR
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor


def from_quat(quat: Array, scalar_first: bool = False) -> R:
    """Creates a rotation object compatible with the type of the given quat."""
    if isinstance(quat, jp.ndarray):
        if scalar_first:
            raise ValueError("scalar_first is not supported by jax rotations")
        return JR.from_quat(quat)
    else:
        return R.from_quat(quat, scalar_first=scalar_first)


def from_rotvec(rotvec: Array, degrees: bool = False) -> R:
    """Creates a rotation object compatible with the type of the given rotvec."""
    if isinstance(rotvec, jp.ndarray):
        return JR.from_rotvec(rotvec, degrees)
    else:
        return R.from_rotvec(rotvec, degrees)


def from_euler(seq: str, angles: Array, degrees: bool = False) -> R:
    """Creates a rotation object compatible with the type of the given angles."""
    if isinstance(angles, jp.ndarray):
        return JR.from_euler(seq, angles, degrees)
    else:
        return R.from_euler(seq, angles, degrees)


def from_matrix(matrix: Array) -> R:
    """Creates a rotation objecte compatible with the type of the given rotation matrix."""
    if isinstance(matrix, jp.ndarray):
        return JR.from_matrix(matrix)
    else:
        return R.from_matrix(matrix)


def quat_mult(q1: Array, q2: Array) -> Array:
    """TODO remove and use scipy implementation."""
    xp = q1.__array_namespace__()
    cross = xp.linalg.cross(q1[..., :3], q2[..., :3])
    qx = q1[..., 3] * q2[..., 0] + q2[..., 3] * q1[..., 0] + cross[..., 0]
    qy = q1[..., 3] * q2[..., 1] + q2[..., 3] * q1[..., 1] + cross[..., 1]
    qz = q1[..., 3] * q2[..., 2] + q2[..., 3] * q1[..., 2] + cross[..., 2]
    qw = (
        q1[..., 3] * q2[..., 3]
        - q1[..., 0] * q2[..., 0]
        - q1[..., 1] * q2[..., 1]
        - q1[..., 2] * q2[..., 2]
    )
    quat = xp.stack([qx, qy, qz, qw], axis=-1)
    return quat


def quat_conj(q: Array) -> Array:
    """TODO remove and use scipy implementation."""
    xp = q.__array_namespace__()
    return xp.concat([-q[..., :3], q[..., 3:]], axis=-1)


def ang_vel2rpy_rates(quat: Array, ang_vel: Array) -> Array:
    """Convert angular velocity to rpy rates with batch support."""
    xp = quat.__array_namespace__()
    rpy = from_quat(quat).as_euler("xyz")
    phi, theta = rpy[..., 0], rpy[..., 1]

    sin_phi = xp.sin(phi)
    cos_phi = xp.cos(phi)
    cos_theta = xp.cos(theta)
    tan_theta = xp.tan(theta)
    inv_cos_theta = 1 / cos_theta

    one = xp.ones_like(phi)
    zero = xp.zeros_like(phi)

    W = xp.stack(
        [
            xp.stack([one, sin_phi * tan_theta, cos_phi * tan_theta], axis=-1),
            xp.stack([zero, cos_phi, -sin_phi], axis=-1),
            xp.stack([zero, sin_phi * inv_cos_theta, cos_phi * inv_cos_theta], axis=-1),
        ],
        axis=-2,
    )

    return xp.matmul(W, ang_vel[..., None])[..., 0]


def rpy_rates2ang_vel(quat: Array, rpy_rates: Array) -> Array:
    """Convert rpy rates to angular velocity with batch support."""
    xp = quat.__array_namespace__()
    rpy = from_quat(quat).as_euler("xyz")
    phi, theta = rpy[..., 0], rpy[..., 1]

    sin_phi = xp.sin(phi)
    cos_phi = xp.cos(phi)
    cos_theta = xp.cos(theta)
    tan_theta = xp.tan(theta)

    one = xp.ones_like(phi)
    zero = xp.zeros_like(phi)

    W = xp.stack(
        [
            xp.stack([one, zero, -cos_theta * tan_theta], axis=-1),
            xp.stack([zero, cos_phi, sin_phi * cos_theta], axis=-1),
            xp.stack([zero, -sin_phi, cos_phi * cos_theta], axis=-1),
        ],
        axis=-2,
    )

    return xp.matmul(W, rpy_rates[..., None])[..., 0]


def ang_vel_deriv2rpy_rates_deriv(quat: Array, ang_vel: Array, ang_vel_deriv: Array) -> Array:
    r"""Convert rpy rates derivatives to angular velocity derivatives.

    .. math::
        \dot{\psi} = \mathbf{\dot{W}}\mathbf{\omega} + \mathbf{W} \dot{\mathbf{\omega}}
    """
    xp = quat.__array_namespace__()
    rpy = from_quat(quat).as_euler("xyz")
    phi, theta = rpy[..., 0], rpy[..., 1]
    rpy_rates = ang_vel2rpy_rates(quat, ang_vel)
    phi_dot, theta_dot = rpy_rates[..., 0], rpy_rates[..., 1]

    sin_phi = xp.sin(phi)
    cos_phi = xp.cos(phi)
    sin_theta = xp.sin(theta)
    cos_theta = xp.cos(theta)
    tan_theta = xp.tan(theta)

    zero = xp.zeros_like(phi)

    W_dot = xp.stack(
        [
            xp.stack(
                [
                    zero,
                    cos_phi * phi_dot * tan_theta + sin_phi * theta_dot / cos_theta**2,
                    -sin_phi * phi_dot * tan_theta + cos_phi * theta_dot / cos_theta**2,
                ],
                axis=-1,
            ),
            xp.stack([zero, -sin_phi * phi_dot, -cos_phi * phi_dot], axis=-1),
            xp.stack(
                [
                    zero,
                    cos_phi * phi_dot / cos_theta + sin_phi * theta_dot * sin_theta / cos_theta**2,
                    -sin_phi * phi_dot / cos_theta + cos_phi * sin_theta * theta_dot / cos_theta**2,
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    return xp.matmul(W_dot, ang_vel[..., None])[..., 0] + ang_vel2rpy_rates(quat, ang_vel_deriv)


def rpy_rates_deriv2ang_vel_deriv(quat: Array, rpy_rates: Array, rpy_rates_deriv: Array) -> Array:
    r"""Convert rpy rates derivatives to angular velocity derivatives.

    .. math::
        \dot{\omega} = \mathbf{\dot{W}}\dot{\mathbf{\psi}} + \mathbf{W} \ddot{\mathbf{\psi}}
    """
    xp = quat.__array_namespace__()
    rpy = from_quat(quat).as_euler("xyz")
    phi, theta = rpy[..., 0], rpy[..., 1]
    phi_dot, theta_dot = rpy_rates[..., 0], rpy_rates[..., 1]

    sin_phi = xp.sin(phi)
    cos_phi = xp.cos(phi)
    sin_theta = xp.sin(theta)
    cos_theta = xp.cos(theta)

    zero = xp.zeros_like(phi)

    W_dot = xp.stack(
        [
            xp.stack([zero, zero, -cos_theta * theta_dot], axis=-1),
            xp.stack(
                [
                    zero,
                    -sin_phi * phi_dot,
                    cos_phi * phi_dot * cos_theta - sin_phi * sin_theta * theta_dot,
                ],
                axis=-1,
            ),
            xp.stack(
                [
                    zero,
                    -cos_phi * phi_dot,
                    -sin_phi * phi_dot * cos_theta - cos_phi * sin_theta * theta_dot,
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    return xp.matmul(W_dot, rpy_rates[..., None])[..., 0] + rpy_rates2ang_vel(quat, rpy_rates_deriv)


def cs_quat2euler(quat: cs.MX, seq: str = "xyz", degrees: bool = False) -> cs.MX:
    """TODO."""
    if len(seq) != 3:
        raise ValueError(f"Expected 3 axes, got {len(seq)}.")

    intrinsic = re.match(r"^[XYZ]{1,3}$", seq) is not None
    extrinsic = re.match(r"^[xyz]{1,3}$", seq) is not None

    if not (intrinsic or extrinsic):
        raise ValueError(
            "Expected axes from `seq` to be from ['x', 'y', 'z'] or ['X', 'Y', 'Z'], got {}".format(
                seq
            )
        )

    if any(seq[i] == seq[i + 1] for i in range(2)):
        raise ValueError("Expected consecutive axes to be different, got {}".format(seq))

    seq = seq.lower()

    # Compute euler from quat
    if extrinsic:
        angle_first = 0
        angle_third = 2
    else:
        seq = seq[::-1]
        angle_first = 2
        angle_third = 0

    def elementary_basis_index(axis: str) -> int:
        """TODO."""
        if axis == "x":
            return 0
        elif axis == "y":
            return 1
        elif axis == "z":
            return 2

    i = elementary_basis_index(seq[0])
    j = elementary_basis_index(seq[1])
    k = elementary_basis_index(seq[2])

    symmetric = i == k

    if symmetric:
        k = 3 - i - j  # get third axis

    # Check if permutation is even (+1) or odd (-1)
    sign = (i - j) * (j - k) * (k - i) // 2

    eps = 1e-7

    if symmetric:
        a = quat[3]
        b = quat[i]
        c = quat[j]
        d = quat[k] * sign
    else:
        a = quat[3] - quat[j]
        b = quat[i] + quat[k] * sign
        c = quat[j] + quat[3]
        d = quat[k] * sign - quat[i]

    angles1 = 2.0 * cs.arctan2(cs.sqrt(c**2 + d**2), cs.sqrt(a**2 + b**2))

    case = cs.if_else(
        cs.fabs(angles1) <= eps, 1, cs.if_else(cs.fabs(angles1 - cs.np.pi) <= eps, 2, 0)
    )

    half_sum = cs.arctan2(b, a)
    half_diff = cs.arctan2(d, c)

    angles_case_0_ = [None, angles1, None]
    angles_case_0_[angle_first] = half_sum - half_diff
    angles_case_0_[angle_third] = half_sum + half_diff
    angles_case_0 = cs.vertcat(*angles_case_0_)

    angles_case_else_ = [None, angles1, 0.0]
    angles_case_else_[0] = cs.if_else(
        case == 1, 2.0 * half_sum, 2.0 * half_diff * (-1.0 if extrinsic else 1.0)
    )
    angles_case_else = cs.vertcat(*angles_case_else_)

    angles = cs.if_else(case == 0, angles_case_0, angles_case_else)

    if not symmetric:
        angles[angle_third] *= sign
        angles[1] -= cs.np.pi * 0.5

    for i in range(3):
        angles[i] += cs.if_else(
            angles[i] < -cs.np.pi,
            2.0 * cs.np.pi,
            cs.if_else(angles[i] > cs.np.pi, -2.0 * cs.np.pi, 0.0),
        )

    if degrees:
        angles = (cs.np.pi / 180.0) * cs.horzcat(angles)

    return angles


def create_cs_ang_vel2rpy_rates() -> cs.Function:
    """TODO."""
    qw = cs.MX.sym("qw")
    qx = cs.MX.sym("qx")
    qy = cs.MX.sym("qy")
    qz = cs.MX.sym("qz")
    quat = cs.vertcat(qx, qy, qz, qw)  # Quaternions
    rpy = cs_quat2euler(quat)
    phi, theta = rpy[0], rpy[1]
    p = cs.MX.sym("p")
    q = cs.MX.sym("q")
    r = cs.MX.sym("r")
    ang_vel = cs.vertcat(p, q, r)  # Angular velocity

    row1 = cs.horzcat(1, cs.sin(phi) * cs.tan(theta), cs.cos(phi) * cs.tan(theta))
    row2 = cs.horzcat(0, cs.cos(phi), -cs.sin(phi))
    row3 = cs.horzcat(0, cs.sin(phi) / cs.cos(theta), cs.cos(phi) / cs.cos(theta))

    W = cs.vertcat(row1, row2, row3)
    rpy_rates = W @ ang_vel

    return cs.Function("cs_ang_vel2rpy_rates", [quat, ang_vel], [rpy_rates])


cs_ang_vel2rpy_rates = create_cs_ang_vel2rpy_rates()


def create_cs_rpy_rates2ang_vel() -> cs.Function:
    """TODO."""
    qw = cs.MX.sym("qw")
    qx = cs.MX.sym("qx")
    qy = cs.MX.sym("qy")
    qz = cs.MX.sym("qz")
    quat = cs.vertcat(qx, qy, qz, qw)  # Quaternions
    rpy = cs_quat2euler(quat)
    phi, theta = rpy[0], rpy[1]
    phi_dot = cs.MX.sym("phi_dot")
    theta_dot = cs.MX.sym("theta_dot")
    psi_dot = cs.MX.sym("psi_dot")
    rpy_rates = cs.vertcat(phi_dot, theta_dot, psi_dot)  # Euler rates

    row1 = cs.horzcat(1, 0, -cs.cos(theta) * cs.tan(theta))
    row2 = cs.horzcat(0, cs.cos(phi), cs.sin(phi) * cs.cos(theta))
    row3 = cs.horzcat(0, -cs.sin(phi), cs.cos(phi) * cs.cos(theta))

    W = cs.vertcat(row1, row2, row3)
    ang_vel = W @ rpy_rates
    return cs.Function("cs_rpy_rates2ang_vel", [quat, rpy_rates], [ang_vel])


cs_rpy_rates2ang_vel = create_cs_rpy_rates2ang_vel()


def create_cs_ang_vel_deriv2rpy_rates_deriv() -> cs.Function:
    """TODO."""
    qw = cs.MX.sym("qw")
    qx = cs.MX.sym("qx")
    qy = cs.MX.sym("qy")
    qz = cs.MX.sym("qz")
    quat = cs.vertcat(qx, qy, qz, qw)  # Quaternions
    rpy = cs_quat2euler(quat)
    phi, theta = rpy[0], rpy[1]
    p = cs.MX.sym("p")
    q = cs.MX.sym("q")
    r = cs.MX.sym("r")
    ang_vel = cs.vertcat(p, q, r)  # Angular velocity
    p_dot = cs.MX.sym("p_dot")
    q_dot = cs.MX.sym("q_dot")
    r_dot = cs.MX.sym("r_dot")
    ang_vel_deriv = cs.vertcat(p_dot, q_dot, r_dot)  # Angular acceleration
    rpy_rates = cs_ang_vel2rpy_rates(quat, ang_vel)
    phi_dot, theta_dot = rpy_rates[0], rpy_rates[1]

    row1 = cs.horzcat(
        0,
        cs.cos(phi) * phi_dot * cs.tan(theta) + cs.sin(phi) * theta_dot / cs.cos(theta) ** 2,
        -cs.sin(phi) * phi_dot * cs.tan(theta) + cs.cos(phi) * theta_dot / cs.cos(theta) ** 2,
    )
    row2 = cs.horzcat(0, -cs.sin(phi) * phi_dot, -cs.cos(phi) * phi_dot)
    row3 = cs.horzcat(
        0,
        cs.cos(phi) * phi_dot / cs.cos(theta)
        + cs.sin(phi) * theta_dot * cs.sin(theta) / cs.cos(theta) ** 2,
        -cs.sin(phi) * phi_dot / cs.cos(theta)
        + cs.cos(phi) * cs.sin(theta) * theta_dot / cs.cos(theta) ** 2,
    )

    W_dot = cs.vertcat(row1, row2, row3)
    rpy_rates_deriv = W_dot @ ang_vel + cs_ang_vel2rpy_rates(quat, ang_vel_deriv)

    return cs.Function("cs_ang_vel2rpy_rates", [quat, ang_vel, ang_vel_deriv], [rpy_rates_deriv])


cs_ang_vel_deriv2rpy_rates_deriv = create_cs_ang_vel_deriv2rpy_rates_deriv()


def create_cs_rpy_rates_deriv2ang_vel_deriv() -> cs.Function:
    """TODO."""
    qw = cs.MX.sym("qw")
    qx = cs.MX.sym("qx")
    qy = cs.MX.sym("qy")
    qz = cs.MX.sym("qz")
    quat = cs.vertcat(qx, qy, qz, qw)  # Quaternions
    rpy = cs_quat2euler(quat)
    phi, theta = rpy[0], rpy[1]
    phi_dot = cs.MX.sym("phi_dot")
    theta_dot = cs.MX.sym("theta_dot")
    psi_dot = cs.MX.sym("psi_dot")
    rpy_rates = cs.vertcat(phi_dot, theta_dot, psi_dot)  # Euler rates
    phi_dot_dot = cs.MX.sym("phi_dot_dot")
    theta_dot_dot = cs.MX.sym("theta_dot_dot")
    psi_dot_dot = cs.MX.sym("psi_dot_dot")
    rpy_rates_deriv = cs.vertcat(phi_dot_dot, theta_dot_dot, psi_dot_dot)  # Euler rates derivative

    row1 = cs.horzcat(0, 0, -cs.cos(theta) * theta_dot)
    row2 = cs.horzcat(
        0,
        -cs.sin(phi) * phi_dot,
        cs.cos(phi) * phi_dot * cs.cos(theta) - cs.sin(phi) * cs.sin(theta) * theta_dot,
    )
    row3 = cs.horzcat(
        0,
        -cs.cos(phi) * phi_dot,
        -cs.sin(phi) * phi_dot * cs.cos(theta) - cs.cos(phi) * cs.sin(theta) * theta_dot,
    )

    W_dot = cs.vertcat(row1, row2, row3)
    ang_vel_deriv = W_dot @ rpy_rates + cs_rpy_rates2ang_vel(quat, rpy_rates_deriv)

    return cs.Function("cs_ang_vel2rpy_rates", [quat, rpy_rates, rpy_rates_deriv], [ang_vel_deriv])


cs_rpy_rates_deriv2ang_vel_deriv = create_cs_rpy_rates_deriv2ang_vel_deriv()


def cs_quat2matrix(quat: cs.MX) -> cs.MX:
    """Creates a symbolic rotation matrix based on a symbolic quaternion.

    From https://github.com/cmower/spatial-casadi/blob/master/spatial_casadi/spatial.py
    """
    x = quat[0] / cs.norm_2(quat)
    y = quat[1] / cs.norm_2(quat)
    z = quat[2] / cs.norm_2(quat)
    w = quat[3] / cs.norm_2(quat)

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = cs.horzcat(
        cs.vertcat(x2 - y2 - z2 + w2, 2.0 * (xy + zw), 2.0 * (xz - yw)),
        cs.vertcat(2.0 * (xy - zw), -x2 + y2 - z2 + w2, 2.0 * (yz + xw)),
        cs.vertcat(2.0 * (xz + yw), 2.0 * (yz - xw), -x2 - y2 + z2 + w2),
    )

    return matrix
