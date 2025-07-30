"""This file is loads all constants for a specific drone, based on the drone type, and stores it in a dataclass."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor


@dataclass
class Constants:
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

    # System Identification parameters for THREE_D_ATTITUDE_DELAY model
    THREE_D_AD_ROLL: Array
    THREE_D_AD_PITCH: Array
    THREE_D_AD_YAW: Array
    THREE_D_AD_PARAMS: Array
    THREE_D_AD_ACC: Array
    THREE_D_AD_TRANSFORM: Array  # [cmd_min, cmd_max, f_min, f_max]

    # Configs (used in testing)
    available_configs: tuple[str] = ("cf2x_L250", "cf2x_P250", "cf2x_T350", "cf2x_L350")

    @classmethod
    def from_file(cls, path: str) -> Constants:
        """Creates constants based on the xml file at the given location.

        The constants are supposed to be under the costum/numeric category.
        """
        # Constants
        drone_path = Path(__file__).parents[1] / path
        # read in all parameters from xml
        params = ET.parse(drone_path).findall(".//custom/numeric")
        # create a dict from parameters containing array of floats
        params = {
            p.get("name"): np.array(list(map(float, p.get("data").split())))
            for p in params
        }

        GRAVITY = params["gravity"][0]
        GRAVITY_VEC = np.array([0, 0, -GRAVITY])
        MASS = params["mass"][0]
        J = params["J"].reshape((3, 3))
        J_INV = np.linalg.inv(J)
        L = params["arm"][0]
        SIGN_MATRIX = params["sign_matrix"].reshape((4, 3))

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
        SI_PARAMS = np.vstack((SI_ROLL, SI_PITCH, SI_YAW))
        SI_ACC = params["SI_acc"]

        # System Identification parameters for the double integrator (DI) model
        DI_ROLL = params["DI_roll"]
        DI_PITCH = params["DI_pitch"]
        DI_YAW = params["DI_yaw"]
        DI_PARAMS = np.vstack((DI_ROLL, DI_PITCH, DI_YAW))
        DI_ACC = params["DI_acc"]

        DI_D_ROLL = params["DI_D_roll"]
        DI_D_PITCH = params["DI_D_pitch"]
        DI_D_YAW = params["DI_D_yaw"]
        DI_D_PARAMS = np.vstack((DI_D_ROLL, DI_D_PITCH, DI_D_YAW))
        DI_D_ACC = params["DI_D_acc"]

        DI_DD_ROLL = params["DI_DD_roll"]
        DI_DD_PITCH = params["DI_DD_pitch"]
        DI_DD_YAW = params["DI_DD_yaw"]
        DI_DD_PARAMS = np.vstack((DI_DD_ROLL, DI_DD_PITCH, DI_DD_YAW))
        DI_DD_ACC = params["DI_DD_acc"]

        # THREE_D_ATTITUDE_DELAY parameters (using defaults if not in XML)
        THREE_D_AD_ROLL = params.get("THREE_D_AD_roll", np.array([-238.1, -21.35, 179.65]))
        THREE_D_AD_PITCH = params.get("THREE_D_AD_pitch", np.array([-238.1, -21.35, 179.65]))  
        THREE_D_AD_YAW = params.get("THREE_D_AD_yaw", np.array([-170.4, -22.22, 280]))
        THREE_D_AD_PARAMS = np.vstack((THREE_D_AD_ROLL, THREE_D_AD_PITCH, THREE_D_AD_YAW))
        THREE_D_AD_ACC = params.get("THREE_D_AD_acc", np.array([-0.04, 0.776, 0.092]))
        THREE_D_AD_TRANSFORM = params.get("THREE_D_AD_transform", np.array([0.2523, 0.4530, 0.2890, 0.4236]))

        return cls(
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
            THREE_D_AD_ROLL,
            THREE_D_AD_PITCH,
            THREE_D_AD_YAW,
            THREE_D_AD_PARAMS,
            THREE_D_AD_ACC,
            THREE_D_AD_TRANSFORM,
        )

    @classmethod
    def from_config(cls, config: str) -> Constants:
        """Creates constants based on the give configuration.

        For available configs see Constants.available_configs
        """
        match config:
            case "cf2x_L250":
                return Constants.from_file("data/cf2x_L250.xml")
            case "cf2x_P250":
                return Constants.from_file("data/cf2x_P250.xml")
            case "cf2x_T350":
                return Constants.from_file("data/cf2x_T350.xml")
            case "cf2x_L350":
                return Constants.from_file("data/cf2x_L350.xml")
            case _:
                raise ValueError(f"Drone config '{config}' is not supported")
