# Drone Models

Drone Models provides several model types with different complexity levels and capabilities. Understanding these differences will help you choose the right model for your application.

## Model Types Overview

| Model               | Rotor Dynamics | Complexity | Physics Level        | Best For                      |
| ------------------- | -------------- | ---------- | -------------------- | ----------------------------- |
| `first_principles`  | Yes            | High       | Full rigid body      | High-fidelity simulation, MPC |
| `so_rpy`            | No             | Low        | Simplified           | Fast simulation, learning     |
| `so_rpy_rotor`      | Yes            | Medium     | Simplified + rotors  | Balanced performance          |
| `so_rpy_rotor_drag` | Yes            | High       | Simplified + effects | High-speed flight             |

## Physics-Based Models

### First Principles Model

The most comprehensive model based on full rigid body dynamics with quaternion representation.

**Features:**

- Quaternion-based attitude representation (no singularities)
- Full 6-DOF rigid body dynamics  
- Rotor spin-up dynamics with time constants
- External force and torque disturbances
- Mixing matrix for arbitrary motor configurations

**When to use:**

- High-fidelity simulations
- Research requiring accurate dynamics
- Hardware-in-the-loop testing

```python
from drone_models.first_principles import dynamics
from drone_models.core import parametrize

# Create parametrized model
model = parametrize(dynamics, "cf2x_L250")

# Use with rotor dynamics
pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = model(
    pos, quat, vel, ang_vel, cmd, rotor_vel=rotor_vel
)
```

**How to obtain:**

- Measure parameters, such as mass
- Obtain thrust curves from experiments
- Identify inertia either from flight or other experiments

### SO(3) Roll-Pitch-Yaw Models

Simplified models using Euler angle representation for faster computation.

#### `so_rpy` - Basic Model
- Direct thrust/torque mapping (no rotor dynamics)
- Euler angle representation
- Fastest execution

#### `so_rpy_rotor` - With Rotor Dynamics  
- Adds rotor spin-up dynamics
- Models motor response delays
- Good balance of speed and accuracy

#### `so_rpy_rotor_drag` - With Aerodynamic Effects
- Includes drag and aerodynamic forces
- Most complete simplified model
- Better at high speeds

**When to use:** Real-time applications requiring speed, e.g., model based control

```python
from drone_models.so_rpy import dynamics as so_rpy_dynamics
from drone_models.so_rpy_rotor import dynamics as so_rpy_rotor_dynamics

# Fast model without rotor dynamics
basic_model = parametrize(so_rpy_dynamics, "cf2x_L250")

# Balanced model with rotor dynamics
balanced_model = parametrize(so_rpy_rotor_dynamics, "cf2x_L250")
```

**How to obtain:**

- Measure parameters, such as mass
- Fit model to a few minutes of flight data

You can use the given pixi environment, which can be activated with `pixi shell -e sysid`. An example is given below:

```python
from drone_models.utils.identification import sys_id_rotation, sys_id_translation

# Your observations as a dict
data = {
    "time": time,
    "pos": pos,
    "quat": quat,
    "cmd_rpy": cmd_rpy,
    "cmd_f": cmd_f,
}

# Preprocessing and derivatives computation 
data = preprocessing(data)
data = derivatives_svf(data)

# Actual sysid scripts
sys_id_translation(model="so_rpy_rotor_drag", mass=mass, data=data)
sys_id_rotation(data=data)
```

For more information on the identification process, check out our [API](../reference/drone_models/index.md)

## Data-Driven Models

While the current focus is on physics-based models, the framework is designed to support data-driven approaches:

**Planned features:**
- Neural network residual models
- Gaussian Process corrections
- System identification tools
- Sim-to-real adaptation

## Model Capabilities

### Rotor Dynamics

Models with rotor dynamics (`first_principles`, `so_rpy_rotor`, `so_rpy_rotor_drag`) include:

- **Motor time constants**: Realistic spin-up/spin-down behavior
- **Command to thrust mapping**: Non-instantaneous response
- **Rotor state tracking**: Additional state variables for rotor velocities

### Symbolic Variants

Each model provides both numeric and symbolic implementations:

```python
# Numeric for simulation
from drone_models.first_principles import dynamics

# Symbolic for optimization
from drone_models.first_principles import symbolic_dynamics

# Generate CasADi function for MPC
X_dot, X, U, Y = symbolic_dynamics(model_rotor_vel=True)
f = cs.Function('dynamics', [X, U], [X_dot])
```

### External Disturbances

All models support external force and torque disturbances:

```python
# Add wind disturbance
wind_force = np.array([2.0, 0.0, 0.0])  # 2 N force in x-direction
turbulence_torque = np.array([0.1, 0.1, 0.0])  # Small torque disturbance

derivatives = model(pos, quat, vel, ang_vel, cmd, rotor_vel,
                   dist_f=wind_force, dist_t=turbulence_torque)
```

## Choosing the Right Model

### For Real-Time Control
- **Fast control loops (>500 Hz)**: `so_rpy`
- **Standard control (100-200 Hz)**: `so_rpy_rotor`
- **High-fidelity control**: `first_principles`

### For Simulation
- **Educational/demos**: `so_rpy` or `so_rpy_rotor`  
- **Research simulation**: `first_principles`
- **Large-scale studies**: `so_rpy` for speed

### For Optimization (MPC)
- **Real-time MPC**: `so_rpy_rotor` symbolic
- **High-fidelity MPC**: `first_principles` symbolic
- **Trajectory optimization**: Any symbolic variant

### For Machine Learning
- **RL training**: `so_rpy` for speed
- **Sim-to-real**: `first_principles` + planned data-driven extensions
- **System identification**: `first_principles` as ground truth

## Performance Considerations

### Computational Cost (relative)
1. `so_rpy`: 1x (baseline)
2. `so_rpy_rotor`: ~1.5x  
3. `so_rpy_rotor_drag`: ~2x
4. `first_principles`: ~3x

### Memory Usage
- All models have minimal memory footprint
- Batching scales linearly with batch size
- GPU memory usage depends on batch size and backend

### Accuracy Trade-offs
- **Highest accuracy**: `first_principles`
- **Good accuracy**: `so_rpy_rotor_drag`
- **Acceptable accuracy**: `so_rpy_rotor`
- **Basic accuracy**: `so_rpy`

---

*Choose your model based on the trade-off between accuracy, computational cost, and your specific application requirements.*
