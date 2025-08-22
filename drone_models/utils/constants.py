"""This file is loads all constants for a specific drone, based on the drone type, and stores it in a dataclass."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from array_api_compat import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    from array_api_typing import Array

# Configs (used in testing)
available_drone_types: tuple = ("cf2x_L250",)  # , "cf2x_P250", "cf2x_T350")


class Constants(NamedTuple):
    """This is a dataclass for all necessary constants in the models."""

    GRAVITY: float
    GRAVITY_VEC: Array
    MASS: float
    J: Array
    J_INV: Array
    L: float
    SIGN_MATRIX: Array

    PWM_MIN: float
    PWM_MAX: float
    KF: float
    KM: float
    THRUST_MIN: float
    THRUST_MAX: float
    THRUST_TAU: float

    # System Identification (SI) parameters
    SI_ROLL: Array
    SI_PITCH: Array
    SI_YAW: Array
    SI_PARAMS: Array
    SI_ACC: Array

    # System Identification parameters for the double integrator (DI) model
    DI_ROLL: Array
    DI_PITCH: Array
    DI_YAW: Array
    DI_PARAMS: Array
    DI_ACC: Array

    # System Identification parameters for the double integrator (DI) model with delay
    DI_D_ROLL: Array
    DI_D_PITCH: Array
    DI_D_YAW: Array
    DI_D_PARAMS: Array
    DI_D_ACC: Array

    DI_DD_ROLL: Array
    DI_DD_PITCH: Array
    DI_DD_YAW: Array
    DI_DD_PARAMS: Array
    DI_DD_ACC: Array

    @staticmethod
    def from_file(path: Path, xp: ModuleType = np) -> Constants:
        """Creates constants based on the xml file at the given location.

        The constants are supposed to be under the costum/numeric category.
        """
        # Constants
        drone_path = Path(__file__).parents[1] / path
        # read in all parameters from xml
        params = ET.parse(drone_path).findall(".//custom/numeric")
        # create a dict from parameters containing array of floats
        params = {
            p.get("name"): xp.asarray(list(map(float, p.get("data").split()))) for p in params
        }

        GRAVITY = params["gravity"][0]
        GRAVITY_VEC = xp.stack([xp.asarray(0.0), xp.asarray(0.0), -GRAVITY])
        MASS = params["mass"][0]
        J = xp.reshape(params["J"], (3, 3))
        J_INV = xp.linalg.inv(J)
        L = params["arm"][0]
        SIGN_MATRIX = xp.reshape(params["sign_matrix"], (4, 3))

        PWM_MIN = params["PWM_MIN"][0]
        PWM_MAX = params["PWM_MAX"][0]
        KF = params["kf"][0]
        KM = params["km"][0]
        THRUST_MIN = params["THRUST_MIN"][0]
        THRUST_MAX = params["THRUST_MAX"][0]
        THRUST_TAU = params["THRUST_TAU"][0]

        # System Identification (SI) parameters
        SI_ROLL = params["SI_roll"]
        SI_PITCH = params["SI_pitch"]
        SI_YAW = params["SI_yaw"]
        SI_PARAMS = xp.stack((SI_ROLL, SI_PITCH, SI_YAW), axis=0)
        SI_ACC = params["SI_acc"]

        # System Identification parameters for the double integrator (DI) model
        DI_ROLL = params["DI_roll"]
        DI_PITCH = params["DI_pitch"]
        DI_YAW = params["DI_yaw"]
        DI_PARAMS = xp.stack((DI_ROLL, DI_PITCH, DI_YAW), axis=0)
        DI_ACC = params["DI_acc"]

        DI_D_ROLL = params["DI_D_roll"]
        DI_D_PITCH = params["DI_D_pitch"]
        DI_D_YAW = params["DI_D_yaw"]
        DI_D_PARAMS = xp.stack((DI_D_ROLL, DI_D_PITCH, DI_D_YAW), axis=0)
        DI_D_ACC = params["DI_D_acc"]

        DI_DD_ROLL = params["DI_DD_roll"]
        DI_DD_PITCH = params["DI_DD_pitch"]
        DI_DD_YAW = params["DI_DD_yaw"]
        DI_DD_PARAMS = xp.stack((DI_DD_ROLL, DI_DD_PITCH, DI_DD_YAW), axis=0)
        DI_DD_ACC = params["DI_DD_acc"]

        return Constants(
            GRAVITY,
            GRAVITY_VEC,
            MASS,
            J,
            J_INV,
            L,
            SIGN_MATRIX,
            PWM_MIN,
            PWM_MAX,
            KF,
            KM,
            THRUST_MIN,
            THRUST_MAX,
            THRUST_TAU,
            SI_ROLL,
            SI_PITCH,
            SI_YAW,
            SI_PARAMS,
            SI_ACC,
            DI_ROLL,
            DI_PITCH,
            DI_YAW,
            DI_PARAMS,
            DI_ACC,
            DI_D_ROLL,
            DI_D_PITCH,
            DI_D_YAW,
            DI_D_PARAMS,
            DI_D_ACC,
            DI_DD_ROLL,
            DI_DD_PITCH,
            DI_DD_YAW,
            DI_DD_PARAMS,
            DI_DD_ACC,
        )

    @staticmethod
    def from_config(config: str, xp: ModuleType = np) -> Constants:
        """Creates constants based on the give configuration.

        For available configs see Constants.available_drone_types.
        """
        xp = np if xp is None else xp
        match config:
            case "cf2x_L250":
                return Constants.from_file("data/cf2x_L250.xml", xp)
            case "cf2x_P250":
                return Constants.from_file("data/cf2x_P250.xml", xp)
            case "cf2x_T350":
                return Constants.from_file("data/cf2x_T350.xml", xp)
            case _:
                raise ValueError(f"Drone config '{config}' is not supported")
