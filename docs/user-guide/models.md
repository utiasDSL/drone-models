# Models

A dynamics model is a function that takes the current state of the drone and a command, and returns the time derivatives of that state. Integrate those derivatives forward and you have a simulation step; evaluate them symbolically and you have an MPC model. The same function serves both purposes.

Every model in this package shares the same state representation:

| Variable | Shape | Description |
|---|---|---|
| `pos` | `(3,)` | Position in world frame [m] |
| `quat` | `(4,)` | Attitude as unit quaternion, scalar-last `xyzw` |
| `vel` | `(3,)` | Linear velocity in world frame [m/s] |
| `ang_vel` | `(3,)` | Angular velocity in body frame [rad/s] |

What differs between models is the command interface, which parameters are needed, and how much physical detail is captured. The table below gives a quick overview — the sections that follow explain each model and when to reach for it.

| Module | `cmd` input | Rotor dynamics | Key added params |
|---|---|---|---|
| `first_principles` | Motor RPMs `(4,)` | Yes | `rpm2thrust`, `rpm2torque`, `mixing_matrix`, `L`, `prop_inertia` |
| `so_rpy_rotor_drag` | rpyt `(4,)` | Yes | `thrust_time_coef`, `drag_matrix` |
| `so_rpy_rotor` | rpyt `(4,)` | Yes | `thrust_time_coef` |
| `so_rpy` | rpyt `(4,)` | No | — |

## first_principles

The full rigid-body physics model. The command is four individual motor RPMs. The model computes forces and torques from the RPMs using polynomial thrust and torque curves, applies the mixing matrix to find body-frame moments, and integrates using Newton–Euler equations. Propeller inertia and gyroscopic effects are included. No fitting to flight data is required — all parameters are physical constants you can measure or look up.

Working at the rotor-velocity level means you need a controller that converts higher-level commands — position setpoints, attitude + collective thrust — down to individual motor RPMs. [drone-controllers](https://utiasdsl.github.io/drone-controllers/) provides a matching set of controllers designed for exactly this interface.

```python
import numpy as np
from drone_models import parametrize
from drone_models.first_principles import dynamics

model = parametrize(dynamics, drone_model="cf2x_L250")

pos, vel, ang_vel = np.zeros((3,)), np.zeros((3,)), np.zeros((3,))
quat = np.array([0., 0., 0., 1.])
cmd = np.full((4,), 15_000.)
rotor_vel = np.full((4,), 12_000.)

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel,
    cmd,        # shape (4,) — motor RPMs
    rotor_vel,  # shape (4,) — current motor RPMs; pass None to skip rotor dynamics
)
```

See the [`first_principles` API reference](../reference/drone_models/first_principles/index.md) for the full parameter list.

## so_rpy_rotor_drag

A fitted second-order model where the command is `[roll_rad, pitch_rad, yaw_rad, thrust_N]` — the same interface used by most flight controller firmware. First-order thrust dynamics model motor spin-up delay, and a linear body-frame drag term accounts for aerodynamic resistance. All coefficients are identified from flight data rather than derived from physics, which makes the model easy to calibrate and well-suited to real-time control.

```{ .python notest }
from drone_models.so_rpy_rotor_drag import dynamics

model = parametrize(dynamics, drone_model="cf2x_L250")

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel,
    cmd,        # shape (4,) — [roll_rad, pitch_rad, yaw_rad, thrust_N]
    rotor_vel,  # shape (1,) — current thrust state [N]; pass None to skip thrust dynamics
)
```

## so_rpy_rotor

The same as `so_rpy_rotor_drag` but without the drag term. Use this when aerodynamic drag is negligible — for example, in low-speed indoor flight — or when you want a slightly simpler model to calibrate.

```python
from drone_models.so_rpy_rotor import dynamics
```

## so_rpy

The simplest model: no rotor dynamics, no drag. The attitude dynamics are a fitted second-order system driven directly by the roll/pitch/yaw command. Passing `rotor_vel` raises a `ValueError`. This model is the fastest to evaluate and the easiest to understand, making it a good baseline for control design and learning-based methods where simulation throughput matters most.

```python
from drone_models.so_rpy import dynamics
```

## External disturbances

All four models accept optional `dist_f` (external force, world frame, N) and `dist_t` (external torque, body frame, N·m) arguments. These are useful for modelling wind, contact forces, or other perturbations without modifying the model itself.

```python
import numpy as np
from drone_models import parametrize
from drone_models.so_rpy import dynamics

model = parametrize(dynamics, drone_model="cf2x_L250")
pos, vel, ang_vel = np.zeros((3,)), np.zeros((3,)), np.zeros((3,))
quat = np.array([0., 0., 0., 1.])
cmd = np.array([0., 0., 0., 0.31])
dist_f = np.array([0.05, 0., 0.])  # 50 mN headwind [N]
dist_t = np.zeros(3)

pos_dot, quat_dot, vel_dot, ang_vel_dot, _ = model(
    pos, quat, vel, ang_vel, cmd, dist_f=dist_f, dist_t=dist_t
)
```

## Checking rotor dynamics support

If you are writing code that works with multiple models, `model_features` tells you programmatically whether a given dynamics function supports rotor dynamics. See the [`model_features` API reference](../reference/drone_models/index.md) for details.

```python
from drone_models import model_features
from drone_models.first_principles import dynamics as fp
from drone_models.so_rpy import dynamics as srpy

model_features(fp)     # {'rotor_dynamics': True}
model_features(srpy)   # {'rotor_dynamics': False}
```

---

With an understanding of the available models, the next step is binding one to a specific drone configuration. That's what [`parametrize`](parametrize.md) does.
