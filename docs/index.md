$$
\dot{x} = f(x, u)
$$

Welcome to **Drone Models** - a Python package providing physics-based and data-driven models of quadrotor drones for estimation, control, and simulation tasks.

`drone-models` provides quadrotor dynamics models as pure Python functions. Models range from full physics-based implementations to lightweight data-driven approximations, each suited to different tasks. All models support NumPy, JAX, and any other [Array API](https://data-apis.org/array-api/latest/) backend, plus symbolic [CasADi](https://web.casadi.org/) variants for optimization-based control. Pre-fitted parameters are included for several Crazyflie platforms.

```python
import numpy as np
from drone_models import parametrize
from drone_models.first_principles import dynamics

model = parametrize(dynamics, drone_model="cf2x_L250")

pos     = np.zeros(3)
quat    = np.array([0., 0., 0., 1.])   # xyzw, identity
vel     = np.zeros(3)
ang_vel = np.zeros(3)
rotor_vel = np.ones(4) * 12_000.
cmd     = np.full(4, 15_000.)           # motor RPMs, near hover

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd, rotor_vel
)
```

## Models

Choosing the right model is a trade-off between accuracy and computational cost. A high-fidelity physics-based model captures rotor spin-up delays, aerodynamic drag, and gyroscopic effects — important for accurate simulation and high-performance control — but is more expensive to evaluate. A lightweight data-driven model can be calibrated from a few minutes of flight data and evaluated in microseconds, making it practical for real-time MPC or large-scale RL training, at the cost of some physical accuracy.

`drone-models` covers this spectrum with multiple models, from first-principles rigid-body dynamics to simple fitted approximations, and is designed to grow with new model types over time. See the [Models](user-guide/models.md) page for a description of each currently available model and when to use it.

## Parametrize

Each model is a plain function with many keyword-only parameters — mass, inertia, thrust curves, and so on. The `parametrize` function binds those parameters to a specific drone configuration, returning a `functools.partial` that you call with just state and command arrays. This keeps the model functions pure and makes it trivial to swap drone configurations or override individual parameters.

Pre-configured parameters are included for several Crazyflie variants, covering the brushed Crazyflie 2.x series and the brushless Crazyflie 2.1. For other drones, you can run the included system identification pipeline to fit the parameters from flight data, and then inject them into any model.

## Backends and batching

All models are built on the [Array API standard](https://data-apis.org/array-api/latest/). This means that the output array type matches whatever you pass in: NumPy arrays in, NumPy arrays out; JAX arrays in, JAX arrays out. Switching backends requires changing a single argument to `parametrize`. No model code changes.

Because the models operate purely on arrays, arbitrary leading batch dimensions work out of the box. Simulate a thousand drones by adding a leading dimension to your state arrays — no loops, no special batch API.

## Symbolic models

Many optimization-based controllers — MPC, trajectory optimization, estimation — require symbolic model representations rather than numeric ones. Every model in this package has a `symbolic_dynamics` function that returns [CasADi](https://web.casadi.org/) `MX` expressions, validated to be numerically equivalent to the numeric implementation. The fitted models also expose `symbolic_dynamics_euler`, which represents attitude as roll/pitch/yaw angles — often the more convenient form for NMPC.

## System identification

The `sysid` module provides a complete pipeline for fitting model parameters from recorded flight data: data preprocessing (outlier removal, interpolation), derivative estimation via a state-variable filter, and least-squares parameter fitting for both the translational and rotational dynamics. Fitting takes a few minutes of flight data and runs in seconds.

## Next steps

- New here? Start with [Installation](get-started/installation.md) and [Quick Start](get-started/quick-start.md).
- See progressively complex usage in [Examples](examples/index.md).
- Ready to dive in? See the [User Guide](user-guide/models.md).
