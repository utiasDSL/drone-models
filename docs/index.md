# Drone Models

$$
\dot{x} = f(x, u)
$$

Welcome to **Drone Models** - a Python package providing physics-based and data-driven models of quadrotor drones for estimation, control, and simulation tasks.

## Overview

Drone Models is a collection of drone dynamics models designed for research and development in autonomous flight. All models are implemented as **pure functions** without internal state, ensuring predictable and reproducible behavior - what you pass in 100% determines what you get out.

## Key Features

### **Multiple Model Types**
- **Physics-based models**: From first principles to simplified dynamics
- **Data-driven models**: For closing the sim-to-real gap
- **Rotor-level dynamics**: Models motor spin-up delays and dynamics
- **Configurable complexity**: Choose the right model for your application

### **Array API Compatibility**
Built on the [Array API standard](https://data-apis.org/array-api/latest/), ensuring **you get the same array type you put in**:

- Pass in NumPy arrays → get NumPy arrays back
- Pass in JAX arrays → get JAX arrays back  
- You get the idea

This means you can:

- **Backpropagate through models** e.g. by using PyTorch tensors
- **JIT compile and auto-differentiate** with JAX, also giving you built-in GPU support
- **Handle arbitrary batch dimensions** - process single drones, batches, or even higher-dimensional arrays seamlessly

### Symbolic Models
While some frameworks like JAX already build a computation graph and allow us to differentiate through the function, many tools from optimization-based control like Acados still require symbolic models from CasADi. We thus also include symbolic CasADi implementations for all models that are tested to be numerically equivalent.

### **High Performance**
- **Batched operations**: Process multiple drones simultaneously
- **JIT compilation**: Fast execution with JAX
- **GPU acceleration**: Scale to large simulations

### **Ready-to-Use**
- **Pre-configured parameters** for Crazyflie 2.x series drones
- **Functional design**: Easy integration with existing codebases
- **Comprehensive testing**: Validated against real hardware
- **MPC-ready**: Direct integration with optimization solvers

## Quick Example

```python
import jax.numpy as jnp
from drone_models import parametrize
from drone_models.first_principles import dynamics

# Create a parametrized model for Crazyflie 2.x
model = parametrize(dynamics, "cf2x_L250", xp=jnp)

# Simulate 100 drones simultaneously
batch_size = 100
pos = jnp.zeros((batch_size, 3))           # Position [m]
quat = jnp.array([[0, 0, 0, 1]] * batch_size)  # Quaternion (xyzw)
vel = jnp.zeros((batch_size, 3))           # Velocity [m/s]
ang_vel = jnp.zeros((batch_size, 3))       # Angular velocity [rad/s]
cmd = jnp.ones((batch_size, 4)) * 1000     # Motor commands [rad/s]

# Calculate dynamics derivatives
pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd, rotor_vel=None
)
```

## Use Cases

This package powers several real-world applications. We use these models in **crazyflow**, our massively parallel, highly accurate drone simulator that can handle millions of drones simultaneously. The models are also integrated into our drone estimators, where the ability to JIT compile everything using JAX ensures low-latency performance critical for real-time operation. Additionally, our MPC implementations rely on this package, taking advantage of the symbolic CasADi models for optimization solvers.

The versatility of the package makes it suitable for simulation applications, control systems design, machine learning research including reinforcement learning and sim-to-real transfer, as well as academic research requiring algorithm validation and hardware-in-the-loop testing.

## Model Portfolio

| Model Type          | Rotor Dynamics | Complexity | Best For                   |
| ------------------- | -------------- | ---------- | -------------------------- |
| `first_principles`  | Yes            | High       | Accurate simulation        |
| `so_rpy`            | No             | Low        | Fast simulation, learning  |
| `so_rpy_rotor`      | Yes            | Medium     | Balanced performance, MPC  |
| `so_rpy_rotor_drag` | Yes            | High       | High-speed flight modeling |

Explore detailed pages for each model: [Models](models/index.md)

## Getting Started

Ready to dive in? Start with our [installation guide](get-started/installation.md) and then try the [quick start tutorial](get-started/quick-start.md).

## Community and Support

- **Documentation**: Comprehensive guides and API reference
- **Issues**: [Report bugs and request features](https://github.com/utiasDSL/drone-models/issues)
- **Discussions**: [Join the community](https://github.com/utiasDSL/drone-models/discussions)
- **License**: MIT License - use it anywhere!

---

*Built with care for the robotics community. Powered by the Array API standard.*
