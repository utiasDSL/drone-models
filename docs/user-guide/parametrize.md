# Parametrize

Each dynamics function has a large number of keyword-only parameters — mass, inertia matrix, thrust curves, drag coefficients, and so on. Passing all of them at every call would be impractical. `parametrize` solves this by loading the parameters for a named drone configuration and returning a [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial) with those parameters pre-filled. You then call the returned object with just the state and command arrays.

```python
from drone_models import parametrize
from drone_models.first_principles import dynamics

model = parametrize(dynamics, drone_model="cf2x_L250")

# Inspect what was pre-filled
list(model.keywords.keys())
# ['mass', 'L', 'prop_inertia', 'gravity_vec', 'J', 'J_inv',
#  'rpm2thrust', 'rpm2torque', 'mixing_matrix', 'rotor_dyn_coef', 'drag_matrix']
```

See the [`parametrize` API reference](../reference/drone_models/index.md) for the full signature.

## Available drone configurations

The following configurations ship with pre-fitted parameters. They cover both the brushed Crazyflie 2.x series and the brushless Crazyflie 2.1:

```python
from drone_models.drones import available_drones

available_drones  # ('cf2x_L250', 'cf2x_P250', 'cf2x_T350', 'cf21B_500')
```

| `drone_model` | Platform |
|---|---|
| `cf2x_L250` | Crazyflie 2.x, L250 props |
| `cf2x_P250` | Crazyflie 2.x, P250 props |
| `cf2x_T350` | Crazyflie 2.x, T350 props |
| `cf21B_500` | Crazyflie 2.1 Brushless, 500 props |

If your drone is not listed, you can identify the parameters from flight data using the [system identification pipeline](system-identification.md) and inject them into any model.

## Switching array backends

By default, `parametrize` stores parameters as NumPy arrays. For frameworks that would otherwise need to convert those arrays on every call — such as PyTorch, where NumPy arrays must become tensors — passing `xp` converts the parameters upfront. The backend of the outputs is always inferred from whatever arrays you pass in at call time.

```python
import torch
import jax.numpy as jnp
from drone_models import parametrize
from drone_models.first_principles import dynamics

model_torch = parametrize(dynamics, drone_model="cf2x_L250", xp=torch)
model_jax   = parametrize(dynamics, drone_model="cf2x_L250", xp=jnp)
```

You can also specify a compute device — for example, to move JAX parameters to GPU at construction time:

```{ .python notest }
import jax
model_gpu = parametrize(
    dynamics, drone_model="cf2x_L250",
    xp=jnp, device=jax.devices("gpu")[0],
)
```

See the [Examples](../examples/index.md#switching-backends) page for a runnable comparison using PyTorch.

## Overriding parameters at call time

Because `parametrize` returns a `functools.partial`, the stored parameters are just keyword argument defaults. You can override any of them for a single call by passing a new value as a keyword argument — the call-time value takes precedence and the stored defaults are unchanged.

```python
from drone_models import parametrize
from drone_models.first_principles import dynamics
import numpy as np
model = parametrize(dynamics, drone_model="cf2x_L250")

pos = np.zeros(3)
quat = np.array([0., 0., 0., 1.])
vel = np.zeros(3)
ang_vel = np.zeros(3)
rotor_vel = np.zeros(4)
cmd = np.zeros(4)

# Simulate with a 10 g payload for this call only — model.keywords is not modified.
pos_dot, *_ = model(pos, quat, vel, ang_vel, cmd, rotor_vel, mass=0.0419)
```

This becomes particularly useful for domain randomization: instead of baking randomized parameters into the partial, you can pass a batch of them as call-time arguments and keep the step function JIT-compiled across parameter changes. See the [domain randomization example](../examples/index.md#domain-randomization-with-jit-compilation) for the full pattern.

## Mutating stored parameters

You can also modify `model.keywords` directly for changes that should persist across all future calls:

```python
from drone_models import parametrize
from drone_models.first_principles import dynamics
import numpy as np
model = parametrize(dynamics, drone_model="cf2x_L250")
model.keywords["mass"] = np.float64(0.040)  # heavier drone — applies to every call
```

!!! warning
    `model.keywords` is a mutable dict shared across all references to the same partial. Modifying it affects every call. Call `parametrize` again for an independent copy.

## Using available_models

`available_models` is a dict mapping model names to their unparametrized dynamics functions. This is useful when selecting a model programmatically.

```python
from drone_models import available_models
from drone_models import parametrize

list(available_models)  # ['first_principles', 'so_rpy', 'so_rpy_rotor', 'so_rpy_rotor_drag']

fn = available_models["so_rpy_rotor_drag"]
model = parametrize(fn, drone_model="cf2x_T350")
```

## Loading raw parameters

If you need the parameter values directly — for example, to pass them to [`symbolic_dynamics`](symbolic.md) — use `load_params`:

```python
from drone_models.core import load_params

params = load_params("first_principles", "cf2x_L250")
params["mass"]        # 0.0319
params["rpm2thrust"]  # array([...])
```

See the [`load_params` API reference](../reference/drone_models/core.md) for details.

---

With a parametrized model in hand, you can evaluate a single state. The next page covers running many drones simultaneously by adding batch dimensions — and how to randomize physical parameters across that batch for domain randomization.
