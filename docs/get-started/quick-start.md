# Quick Start

This page walks through a complete minimal workflow with the first-principles model: bind parameters to a drone, set up a state, and evaluate the state derivatives.

## State and command

Every model takes the same state representation:

| Variable | Description | Shape | Convention |
|---|---|---|---|
| `pos` | Position in world frame | `(3,)` | metres, NED or ENU depending on your setup |
| `quat` | Attitude as unit quaternion | `(4,)` | scalar-last `xyzw` |
| `vel` | Linear velocity in world frame | `(3,)` | m/s |
| `ang_vel` | Angular velocity in body frame | `(3,)` | rad/s |

Quaternions are used instead of Euler angles because they have no singularities — no gimbal lock at 90° pitch — which matters when a drone tilts aggressively.

The command `cmd` depends on the model. For `first_principles` it is a `(4,)` array of motor angular velocities in RPM, one per motor.

## Evaluate dynamics

`parametrize` loads the physical parameters for a specific drone configuration and returns a function you can call with just the state and command. Here we use the Crazyflie 2.x with L250 propellers.

```python
import numpy as np
from drone_models import parametrize
from drone_models.first_principles import dynamics

# Bind physical parameters — returns a functools.partial with kwargs pre-filled.
model = parametrize(dynamics, drone_model="cf2x_L250")

# State: hovering at origin, upright, stationary.
pos     = np.zeros(3)                   # [m]
quat    = np.array([0., 0., 0., 1.])   # xyzw — identity (no rotation)
vel     = np.zeros(3)                   # [m/s]
rotor_vel = np.ones(4) * 12_000.        # [RPM] — motors are spinning but not yet at the 15 000 RPM
ang_vel = np.zeros(3)                   # [rad/s]

# Command: all four motors at 15 000 RPM (rough hover point for cf2x_L250).
cmd = np.full(4, 15_000.)              # [RPM]

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd, rotor_vel
)
```

## Outputs

The model returns state derivatives — the time derivative of each state variable:

| Return value | Description | Shape | Units |
|---|---|---|---|
| `pos_dot` | Linear velocity (equal to `vel`) | `(3,)` | m/s |
| `quat_dot` | Quaternion rate | `(4,)` | 1/s |
| `vel_dot` | Linear acceleration | `(3,)` | m/s² |
| `ang_vel_dot` | Angular acceleration | `(3,)` | rad/s² |
| `rotor_vel_dot` | Motor RPM rate of change | `(4,)` or `None` | RPM/s |

These are the right-hand side of the continuous-time ODE $\dot{x} = f(x, u)$. To simulate forward in time, integrate them with any ODE integrator (e.g. `scipy.integrate.solve_ivp` or a simple Euler step).

`rotor_vel_dot` is `None` when `rotor_vel` is not provided. In that case the model treats the commanded RPM as the instantaneous motor state — there is no spin-up lag.

## Including rotor dynamics

Real motors don't respond instantaneously to commands. Passing the current motor state as `rotor_vel` enables the rotor dynamics model, which computes how the motors accelerate or decelerate toward the commanded RPM.

```python
import numpy as np
from drone_models import parametrize
from drone_models.first_principles import dynamics

model = parametrize(dynamics, drone_model="cf2x_L250")
pos     = np.zeros(3)
quat    = np.array([0., 0., 0., 1.])
vel     = np.zeros(3)
ang_vel = np.zeros(3)
cmd     = np.full(4, 15_000.)
# Current RPMs lag behind the 15 000 RPM command — motors are still spinning up.
rotor_vel = np.full(4, 12_000.)

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd, rotor_vel=rotor_vel
)
rotor_vel_dot  # positive — rotors accelerating toward cmd
```

## Next steps

- [Models](../user-guide/models.md) — all available models and their command interfaces
- [Parametrize](../user-guide/parametrize.md) — switching drone configurations and array backends
- [Batching and Domain Randomization](../user-guide/batching.md) — vectorized evaluation over many drones
- [Symbolic Models](../user-guide/symbolic.md) — CasADi variants for MPC and optimization
- [System Identification](../user-guide/system-identification.md) — fitting parameters from flight data
