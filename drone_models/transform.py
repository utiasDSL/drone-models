"""Transformations between physical parameters of the quadrotors.

Conversions such as from motor forces to rotor speeds, or from thrust to PWM, are bundled in this
module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

if TYPE_CHECKING:
    from array_api_typing import Array


def motor_force2rotor_speed(motor_forces: Array, kf: float | Array) -> Array:
    """Convert motor forces to rotor speeds.

    Args:
        motor_forces: Motor forces in SI units with shape (..., N).
        kf: Thrust coefficient.

    Returns:
        Array of rotor speeds in rad/s with shape (..., N).
    """
    xp = array_namespace(motor_forces)
    return xp.sqrt(motor_forces / kf)


def force2pwm(thrust: Array | float, thrust_max: Array | float, pwm_max: Array | float) -> Array:
    """Convert thrust in N to thrust in PWM.

    Args:
        thrust: Array or float of the thrust in [N]
        thrust_max: Maximum thrust in [N]
        pwm_max: Maximum PWM value

    Returns:
        Thrust converted in PWM.
    """
    return thrust / thrust_max * pwm_max


def pwm2force(
    pwm: Array | float, thrust_max: Array | float, pwm_max: Array | float
) -> Array | float:
    """Convert pwm thrust command to actual thrust.

    Args:
        pwm: Array or float of the pwm value
        thrust_max: Maximum thrust in [N]
        pwm_max: Maximum PWM value

    Returns:
        thrust: Array or float thrust in [N]
    """
    return pwm / pwm_max * thrust_max
