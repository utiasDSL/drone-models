$$
\huge \displaystyle \dot{x} = f(x,u)
$$

---

Physics-based and data-driven quadrotor dynamics models for estimation, control, and simulation.

[![Python Version]][Python Version URL] [![Ruff Check]][Ruff Check URL] [![Tests]][Tests URL] [![Docs]][Docs URL]

[Python Version]: https://img.shields.io/badge/python-3.10+-blue.svg
[Python Version URL]: https://www.python.org

[Ruff Check]: https://github.com/utiasDSL/drone-models/actions/workflows/ruff.yml/badge.svg?style=flat-square
[Ruff Check URL]: https://github.com/utiasDSL/drone-models/actions/workflows/ruff.yml

[Tests]: https://github.com/utiasDSL/drone-models/actions/workflows/testing.yml/badge.svg
[Tests URL]: https://github.com/utiasDSL/drone-models/actions/workflows/testing.yml

[Docs]: https://github.com/utiasDSL/drone-models/actions/workflows/docs.yml/badge.svg
[Docs URL]: https://utiasdsl.github.io/drone-models/

## Overview

`drone-models` provides quadrotor dynamics as pure Python functions, from full physics-based implementations to lightweight data-driven approximations. All models support NumPy, JAX, and any [Array API](https://data-apis.org/array-api/latest/) backend, plus [CasADi](https://web.casaid.org/) symbolic variants for optimization-based control. Pre-fitted parameters are included for several Crazyflie platforms.

**Available models** — ranging from high-fidelity to lightweight:

| Model | Description |
|---|---|
| `first_principles` | Full rigid-body physics with optional rotor dynamics and aerodynamic drag |
| `so_rpy_rotor_drag` | Data-driven, attitude as roll/pitch/yaw, with rotor dynamics and drag |
| `so_rpy_rotor` | Data-driven, attitude as roll/pitch/yaw, with rotor dynamics |
| `so_rpy` | Lightest data-driven model, attitude as roll/pitch/yaw, no rotor dynamics |

**Pre-fitted configurations** for Crazyflie 2.x (brushed) and Crazyflie 2.1 Brushless:
`cf2x_L250`, `cf2x_P250`, `cf2x_T350`, `cf21B_500`

## Installation

```bash
pip install drone-models
```

For system identification (fitting parameters from your own flight data):

```bash
pip install "drone-models[sysid]"
```

> **Note:** `drone_models` must be imported before SciPy to enable Array API support. If you encounter a `RuntimeError`, either import `drone_models` first or set `export SCIPY_ARRAY_API=1` in your shell.

## Usage

### Basic

Bind parameters to a drone configuration with `parametrize`, then call the model with state and command arrays:

```python
import numpy as np
from drone_models import parametrize
from drone_models.first_principles import dynamics

model = parametrize(dynamics, drone_model="cf2x_L250")

pos     = np.zeros(3)
quat    = np.array([0., 0., 0., 1.])   # xyzw, identity
vel     = np.zeros(3)
ang_vel = np.zeros(3)
rotor_vel = np.ones(4) * 12_000.        # current motor RPMs
cmd     = np.full(4, 15_000.)           # commanded motor RPMs

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd, rotor_vel
)
```

The model returns continuous-time state derivatives $\dot{x} = f(x, u)$. Integrate with any ODE solver (e.g. `scipy.integrate.solve_ivp`) to simulate forward in time.

### Switching backends

Pass any Array API-compatible array and the output type follows automatically — no code changes needed:

```python
import jax.numpy as jnp
from drone_models import parametrize
from drone_models.first_principles import dynamics

model = parametrize(dynamics, drone_model="cf2x_L250", xp=jnp)
# Pass jax arrays — get jax arrays back
```

Arbitrary leading batch dimensions work out of the box: stack states for a thousand drones and evaluate them in one call.

### Symbolic models (CasADi)

Every model exposes a `symbolic_dynamics` function returning CasADi `MX` expressions, for use in MPC, trajectory optimization, or estimation:

```python
from drone_models import parametrize
from drone_models.first_principles import symbolic_dynamics

sym_model = parametrize(symbolic_dynamics, drone_model="cf2x_L250")
```

## Development

Clone and install with [pixi](https://pixi.sh):

```bash
git clone https://github.com/utiasDSL/drone-models.git
cd drone-models
pixi install
```

Run tests:

```bash
pixi run -e tests tests
```

## Citation

Citation information coming soon. See the [docs](https://utiasdsl.github.io/drone-models/) for updates.
