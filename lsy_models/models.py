"""This file returns a selected model with the selected config."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import casadi as cs
import numpy as np

from lsy_models import (
    controllers_numeric,
    models_numeric,  # TODO would be nice to directly import all the functions -> currently they are called the same though
    models_symbolic,
)
from lsy_models.utils.constants import Constants

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor

# Used in testing
available_models = {
    # "model_name": [motor_dynamics, forces_dist, torques_dist],
    "first_principles": [True, True, True],
    "fitted_DI_rpyt": [False, True, True],
    "fitted_DI_D_rpyt": [True, True, True],
    "three_d_attitude_delay": [True, True, True],
}
#  TODO "mellinger_rpyt",


def dynamics_numeric(
    model: str, config: str
) -> Callable[
    [Array, Array, Array, Array, Array, Array, Array | None, Array | None],
    tuple[Array, Array, Array, Array, Array | None],
]:
    """Creates a numerical dynamics function f(x,u).

    Args:
        model: The chosen dynamical model. See available_models
        config: The chosen config/constants for the model. See Constants.available_configs

    Returns:
        A callable which takes the states and input and returns the derivative of the state.

    Warning:
        Do not use quat_dot directly for integration! Only usage of ang_vel is mathematically correct.
        If you still decide to use quat_dot to integrate, ensure unit length!
        More information https://ahrs.readthedocs.io/en/latest/filters/angular.html
    """
    constants = Constants.from_config(config)

    match model:
        case "first_principles":
            return partial(models_numeric.f_first_principles, constants=constants)
        case "mellinger_rpyt":

            def control_plus_model(
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
                command_rpyt = command.copy()
                command_rpyt[..., :-1] = command[..., :-1] * 180 / np.pi  # rad 2 deg
                forces, _ = controllers_numeric.cntrl_mellinger_attitude(
                    pos, quat, vel, ang_vel, command_rpyt, constants
                )
                return models_numeric.f_first_principles(
                    pos,
                    quat,
                    vel,
                    ang_vel,
                    forces,
                    constants,
                    forces_motor,
                    forces_dist,
                    torques_dist,
                )

            return partial(control_plus_model, constants=constants)

        case "fitted_SI":
            raise ValueError(f"Model '{model}' is not supported")  # TODO
        case "fitted_DI_rpyt":
            return partial(models_numeric.f_fitted_DI_rpyt, constants=constants)
        case "fitted_DI_D_rpyt":
            return partial(models_numeric.f_fitted_DI_D_rpyt, constants=constants)
        case "fitted_DI_DD_rpyt":
            return partial(models_numeric.f_fitted_DI_DD_rpyt, constants=constants)
        case "three_d_attitude_delay":
            return partial(models_numeric.f_three_d_attitude_delay, constants=constants)
        case _:
            raise ValueError(f"Model '{model}' is not supported")


def dynamic_numeric_from_symbolic(
    model: str, config: str
) -> Callable[
    [Array, Array, Array, Array, Array, Array, Array | None, Array | None],
    tuple[Array, Array, Array, Array, Array | None],
]:
    """Creates a numerical dynamics function f(x,u) from a CasADi model.

    Args:
        model: The chosen dynamical model. See available_models
        config: The chosen config/constants for the model. See Constants.available_configs

    Returns:
        A callable which takes the states and input and returns the derivative of the state.
    """
    constants = Constants.from_config(config)

    match model:
        case "first_principles":
            X_dot, X, U, _ = models_symbolic.first_principles(constants)
            return cs.Function("first_principles", [X, U], [X_dot])
        case "fitted_SI_rpyt":
            raise ValueError(f"Model '{model}' is not supported")  # TODO
        case "fitted_DI_rpyt":
            X_dot, X, U, _ = models_symbolic.f_fitted_DI_rpyt(constants)
            return cs.Function("fitted_DI_rpyt", [X, U], [X_dot])
        case "fitted_DI_D_rpyt":
            X_dot, X, U, _ = models_symbolic.f_fitted_DI_D_rpyt(constants)
            return cs.Function("fitted_DI_D_rpyt", [X, U], [X_dot])
        case "three_d_attitude_delay":
            X_dot, X, U, _ = models_symbolic.three_d_attitude_delay(constants)
            return cs.Function("three_d_attitude_delay", [X, U], [X_dot])
        case _:
            raise ValueError(f"Model '{model}' is not supported")


def observation_function(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    command: Array,
    forces_motor: Array | None = None,
    forces_dist: Array | None = None,
    torques_dist: Array | None = None,
) -> Array:
    """Return the observable part of the state.

    This is basically not necessary, since we always get position and orientation
    from Vicon. However, for sake of completeness, this observation function is added.
    """
    xp = pos.__array_namespace__()
    return xp.concat((pos, quat), axis=-1)


# TODO method for a casadi optimizer object
