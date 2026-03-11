# Examples

These examples build on each other — each one introduces one new idea on top of the previous. Start from the top if you're new, or jump to whichever section covers what you need.

## Single evaluation with NumPy

Everything starts here. `parametrize` loads the physical parameters for a specific drone and returns a function you call with just the state and command arrays. The drone is hovering at the origin — zero position, identity quaternion, no velocity — and all four motors are commanded to 15 000 RPM.

```python
import numpy as np
from drone_models import parametrize
from drone_models.first_principles import dynamics

model = parametrize(dynamics, drone_model="cf2x_L250")

pos     = np.zeros(3)
quat    = np.array([0., 0., 0., 1.])   # xyzw — identity (no rotation)
vel     = np.zeros(3)
ang_vel = np.zeros(3)
rotor_vel = np.ones(4) * 12_000.
cmd     = np.full(4, 15_000.)           # motor RPMs

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd, rotor_vel
)
```

The outputs are the time derivatives of each state variable — the right-hand side of $\dot{x} = f(x, u)$. Pass them to any ODE integrator to simulate forward in time.

## Rotor dynamics

In the previous example `rotor_vel_dot` is `None` because we didn't tell the model what speed the motors are currently at. The model just assumed they were already at the commanded RPM. That's a reasonable approximation for slow maneuvers, but real motors take time to spin up and down. Passing `rotor_vel` enables the rotor dynamics, which computes the acceleration of each motor toward its target.

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
# The motors are at 12 000 RPM but commanded to 15 000 — they're spinning up.
rotor_vel = np.full(4, 12_000.)

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd, rotor_vel=rotor_vel
)
rotor_vel_dot  # positive — rotors accelerating toward cmd
```

## Fitted models

The first-principles model requires individual motor RPMs as input, which means you need rotor-level commands. The fitted models — `so_rpy`, `so_rpy_rotor`, `so_rpy_rotor_drag` — take a higher-level command instead: roll, pitch, yaw setpoints in radians plus collective thrust in Newtons. This matches the command interface of typical flight controllers and makes them convenient for control design and system identification.

```python
import numpy as np
from drone_models import parametrize
from drone_models.so_rpy import dynamics

pos     = np.zeros(3)
quat    = np.array([0., 0., 0., 1.])
vel     = np.zeros(3)
ang_vel = np.zeros(3)

model = parametrize(dynamics, drone_model="cf2x_L250")

# Collective thrust near hover: mass * g ≈ 0.0319 * 9.81 ≈ 0.31 N
cmd = np.array([0., 0., 0., 0.31])   # [roll_rad, pitch_rad, yaw_rad, thrust_N]

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd
)
```

## Switching backends

The models work with any framework that implements the [Array API standard](https://data-apis.org/array-api/latest/). The backend is inferred from the arrays you pass in at call time — you always get the same type back. For frameworks like PyTorch, where the stored NumPy parameters would need to be converted to tensors on every call, passing `xp` to `parametrize` converts them upfront and avoids that overhead.

```python
import torch
from drone_models import parametrize
from drone_models.first_principles import dynamics

# Parameters are stored as torch tensors — no per-call conversion needed.
model = parametrize(dynamics, drone_model="cf2x_L250", xp=torch)

pos     = torch.zeros(3)
quat    = torch.tensor([0., 0., 0., 1.])
vel     = torch.zeros(3)
ang_vel = torch.zeros(3)
rotor_vel = torch.ones(4) * 12_000.
cmd     = torch.full((4,), 15_000.)

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd, rotor_vel
)
```

## Batched evaluation

The same model handles arbitrary leading batch dimensions — no special API, no loops. Add a leading dimension to all state and command arrays and the model evaluates all instances in a single call. This works identically across all backends.

```python
import torch

from drone_models import parametrize
from drone_models.first_principles import dynamics

# Parameters are stored as torch tensors — no per-call conversion needed.
model = parametrize(dynamics, drone_model="cf2x_L250", xp=torch)

N = 1_000

pos       = torch.zeros(N, 3)
quat      = torch.tensor([0., 0., 0., 1.]).expand(N, 4)
vel       = torch.zeros(N, 3)
ang_vel   = torch.zeros(N, 3)
cmd       = torch.full((N, 4), 15_000.)
rotor_vel = torch.full((N, 4), 15_000.)

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd, rotor_vel=rotor_vel
)
vel_dot.shape   # (1000, 3)
```

## Overriding parameters at call time

`parametrize` returns a `functools.partial`, which means the physical parameters it binds are just keyword argument defaults. You can override any of them by passing a different value at call time — the call-time value takes precedence. This is useful for quick experiments without re-parametrizing, and it's the foundation for the domain randomization pattern in the next example.

Here we switch to JAX, which we'll use for the remaining examples.

```python
import jax.numpy as jnp
from drone_models import parametrize
from drone_models.first_principles import dynamics

model = parametrize(dynamics, drone_model="cf2x_L250", xp=jnp)

pos     = jnp.zeros(3)
quat    = jnp.array([0., 0., 0., 1.])
vel     = jnp.zeros(3)
ang_vel = jnp.zeros(3)
rotor_vel = jnp.ones(4) * 12_000.
cmd     = jnp.full((4,), 15_000.)

# The model was parametrized with mass=0.0319 kg.
# Simulate the same drone carrying a 10 g payload for this one call:
pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd, rotor_vel, mass=jnp.float32(0.0419)
)
```

The stored parameters in `model.keywords` are not changed — the override only applies to this call.

## Domain randomization with JIT compilation

Overriding parameters at call time becomes especially powerful for domain randomization. Instead of baking a fixed set of randomized parameters into the function, pass them as explicit arguments to a JIT-compiled step function. JAX traces them as inputs rather than capturing them as constants, so you can draw a fresh batch of parameters each rollout without triggering a recompile.

```python
import jax
import jax.numpy as jnp
from drone_models import parametrize
from drone_models.first_principles import dynamics

N   = 4_096
key = jax.random.PRNGKey(0)

model = parametrize(dynamics, drone_model="cf2x_L250", xp=jnp)

# Batch state and commands
pos       = jnp.zeros((N, 3))
quat      = jnp.tile(jnp.array([0., 0., 0., 1.]), (N, 1))
vel       = jnp.zeros((N, 3))
ang_vel   = jnp.zeros((N, 3))
cmd       = jnp.full((N, 4), 15_000.)
rotor_vel = jnp.full((N, 4), 15_000.)

# mass, J, and J_inv are traced arguments — varying them per call costs no recompilation.
@jax.jit
def step(pos, quat, vel, ang_vel, cmd, rotor_vel, mass, J, J_inv):
    return model(
        pos, quat, vel, ang_vel, cmd, rotor_vel=rotor_vel,
        mass=mass, J=J, J_inv=J_inv,
    )

# Sample randomized parameters for this rollout — ±10% around nominal
nominal_mass = model.keywords["mass"]   # scalar
nominal_J    = model.keywords["J"]      # (3, 3)

key, k1, k2 = jax.random.split(key, 3)
mass_batch  = nominal_mass * jax.random.uniform(k1, (N, 1),      minval=0.9, maxval=1.1)
J_batch     = nominal_J    * jax.random.uniform(k2, (N, 3, 3), minval=0.9, maxval=1.1)
J_inv_batch = jnp.linalg.inv(J_batch)

# First call compiles; subsequent calls with new parameters skip recompilation.
pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = step(
    pos, quat, vel, ang_vel, cmd, rotor_vel,
    mass_batch, J_batch, J_inv_batch,
)
vel_dot.shape   # (4096, 3)
```
