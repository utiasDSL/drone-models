"""Core tools for registering and capability checking for the drone models."""

from __future__ import annotations

import inspect
import tomllib
import warnings
from functools import partial, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    from drone_models._typing import Array  # To be changed to a Protocol later (see array-api#589)


F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
R = TypeVar("R")


def supports(rotor_dynamics: bool = True) -> Callable[[F], F]:
    """Decorator to indicate which features are supported."""

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(
            pos: Array,
            quat: Array,
            vel: Array,
            ang_vel: Array,
            cmd: Array,
            rotor_vel: Array | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[Array, Array, Array, Array, Array | None]:
            if not rotor_dynamics and rotor_vel is not None:
                raise ValueError("Rotor dynamics not supported, but rotor_vel is provided.")
            if rotor_dynamics and rotor_vel is None:
                warnings.warn("Rotor velocity not provided, using commanded rotor velocity.")
            return fn(pos, quat, vel, ang_vel, cmd, rotor_vel, *args, **kwargs)

        wrapper.__drone_model_features__ = {"rotor_dynamics": rotor_dynamics}

        return wrapper  # type: ignore

    return decorator


def parametrize(
    fn: Callable[P, R], drone_model: str, xp: ModuleType | None = None, device: str | None = None
) -> Callable[P, R]:
    """Parametrize a dynamics function with the default dynamics parameters for a drone model.

    Args:
        fn: The dynamics function to parametrize.
        drone_model: The drone model to use.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If none, the device is inferred from the xp module.

    Example:
        ```python
        from drone_models.core import parametrize
        from drone_models.first_principles import dynamics

        dynamics_fn = parametrize(dynamics, drone_model="cf2x_L250")
        pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = dynamics_fn(
            pos=pos, quat=quat, vel=vel, ang_vel=ang_vel, cmd=cmd, rotor_vel=rotor_vel
        )
        ```

    Returns:
        The parametrized dynamics function with all keyword argument only parameters filled in.
    """
    try:
        xp = np if xp is None else xp
        physics = fn.__module__.split(".")[-2]
        sig = inspect.signature(fn)
        kwonly_params = [
            name
            for name, param in sig.parameters.items()
            if param.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        params = load_params(physics, drone_model)
        params = {k: xp.asarray(v, device=device) for k, v in params.items() if k in kwonly_params}
    except KeyError as e:
        raise KeyError(
            f"Model `{physics}` does not exist in the parameter registry for drone `{drone_model}`"
        ) from e
    except ValueError as e:
        raise ValueError(f"Drone model `{drone_model}` not supported for `{physics}`") from e
    return partial(fn, **params)


def load_params(physics: str, drone_model: str, xp: ModuleType | None = None) -> dict:
    """TODO.

    Args:
        physics: _description_
        drone_model: _description_
        xp: The array API module to use. If not provided, numpy is used.

    Returns:
        dict[str, Array]: _description_
    """
    xp = np if xp is None else xp
    with open(Path(__file__).parent / "data/params.toml", "rb") as f:
        physical_params = tomllib.load(f)
    if drone_model not in physical_params:
        raise KeyError(f"Drone model `{drone_model}` not found in data/params.toml")
    with open(Path(__file__).parent / f"{physics}/params.toml", "rb") as f:
        model_params = tomllib.load(f)
    if drone_model not in model_params:
        raise KeyError(f"Drone model `{drone_model}` not found in model params.toml")
    params = physical_params[drone_model] | model_params[drone_model]
    params["J_inv"] = np.linalg.inv(params["J"])
    params = {k: xp.asarray(v) for k, v in params.items()}  # if k in fields
    return params
