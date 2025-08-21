"""Implementations of onboard drone controllers in Python.

All controllers are implemented using the array API standard. This means that every controller is
agnostic to the choice of framework and supports e.g. NumPy, JAX, or PyTorch. We also implement all
controllers as pure functions to ensure that users can jit-compile them. All controllers use
broadcasting to support batching of arbitrary leading dimensions.
"""

from typing import Callable

from drone_models.controller.core import parametrize
from drone_models.controller.mellinger import (
    attitude2force_torque as mellinger_attitude2force_torque,
)
from drone_models.controller.mellinger import state2attitude as mellinger_state2attitude

available_controller: dict[str, Callable] = {
    "mellinger_state2attitude": mellinger_state2attitude,
    "mellinger_attitude2force_torque": mellinger_attitude2force_torque,
}

__all__ = ["parametrize"]
