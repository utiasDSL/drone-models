# System Identification

If your drone is not among the [supported configurations](../get-started/installation.md#supported-drone-configurations), or if you want to refine the existing parameters with your own hardware, the `sysid` pipeline fits the model coefficients from recorded flight data. It handles data preprocessing, derivative estimation, and least-squares parameter fitting for both translational and rotational dynamics.

This requires the `sysid` extra:

```bash
pip install "drone-models[sysid]"
# or, with pixi:
pixi shell -e sysid
```

## Required data format

The pipeline expects a Python dict of NumPy arrays assembled from your flight log. The keys below are required by [`preprocessing`](../reference/drone_models/utils/data_utils/):

| Key | Shape | Units | Description |
|---|---|---|---|
| `"time"` | `(N,)` | s | Timestamps (need not be evenly spaced) |
| `"pos"` | `(N, 3)` | m | Position in world frame |
| `"quat"` | `(N, 4)` | — | Orientation quaternion (xyzw) |
| `"cmd_rpy"` | `(N, 3)` | rad | Commanded roll/pitch/yaw |
| `"cmd_f"` | `(N,)` | N | Commanded collective thrust |

After `preprocessing` + [`derivatives_svf`](../reference/drone_models/utils/data_utils/), the dict is augmented with filtered signals and numerical derivatives. The identification functions read `SVF_vel`, `SVF_acc`, `SVF_quat`, `SVF_cmd_f` (translation) and `SVF_rpy`, `SVF_cmd_rpy` (rotation).

## Full pipeline

```python
from drone_models.utils.data_utils import preprocessing, derivatives_svf
from drone_models.utils.identification import sys_id_translation, sys_id_rotation

# Step 1 — assemble raw data dict from your flight log
data = {
    "time":    time_array,     # (N,) seconds
    "pos":     pos_array,      # (N, 3) metres
    "quat":    quat_array,     # (N, 4) xyzw
    "cmd_rpy": cmd_rpy_array,  # (N, 3) radians
    "cmd_f":   cmd_f_array,    # (N,)  Newtons
}

# Step 2 — outlier removal, quaternion normalisation, RPY calculation
data = preprocessing(data)

# Step 3 — low-pass filter and compute time derivatives via State Variable Filter
data = derivatives_svf(data)

# Step 4 — fit translational parameters
trans_params = sys_id_translation(
    model="so_rpy_rotor_drag",
    mass=0.0319,     # drone mass in kg — measure this directly
    data=data,
    verbose=0,       # 0 = silent, 1 = progress, 2 = full optimizer output
    plot=True,       # show fit vs. measured plots
)
# Returns: {'cmd_f_coef': ..., 'thrust_time_coef': ...,
#           'drag_xy_coef': ..., 'drag_z_coef': ...}

# Step 5 — fit rotational parameters
rot_params = sys_id_rotation(data=data, verbose=0, plot=True)
# Returns: {'rpy_coef': (3,), 'rpy_rates_coef': (3,), 'cmd_rpy_coef': (3,)}
```

See the [`sys_id_translation`](../reference/drone_models/utils/identification/) and [`sys_id_rotation`](../reference/drone_models/utils/identification/) API references for the full argument list.

## Validation split

To check that the identified parameters generalise beyond the training data, provide a held-out split as `data_validation`. RMSE and R² are then reported on both sets.

```python
n = int(0.8 * len(time_array))
# preprocessing must be called on each split independently
data_train = preprocessing({k: v[:n] for k, v in raw_data.items()})
data_valid = preprocessing({k: v[n:] for k, v in raw_data.items()})
data_train = derivatives_svf(data_train)
data_valid = derivatives_svf(data_valid)

trans_params = sys_id_translation(
    model="so_rpy_rotor_drag",
    mass=0.0319,
    data=data_train,
    data_validation=data_valid,
    plot=True,
)
```

## Using identified parameters

Once you have the identified coefficients, inject them into a parametrized model by overriding the relevant entries in `model.keywords`. See the [Parametrize](parametrize.md#mutating-stored-parameters) page for more on this pattern.

```python
import numpy as np
from drone_models import parametrize
from drone_models.so_rpy_rotor_drag import dynamics

model = parametrize(dynamics, drone_model="cf2x_L250")

model.keywords["cmd_f_coef"]       = trans_params["cmd_f_coef"]
model.keywords["thrust_time_coef"] = trans_params["thrust_time_coef"]
model.keywords["rpy_coef"]         = rot_params["rpy_coef"]
model.keywords["rpy_rates_coef"]   = rot_params["rpy_rates_coef"]
model.keywords["cmd_rpy_coef"]     = rot_params["cmd_rpy_coef"]

# drag_xy_coef and drag_z_coef are scalars — assemble the matrix manually
drag_xy = trans_params["drag_xy_coef"]
drag_z  = trans_params["drag_z_coef"]
model.keywords["drag_matrix"] = np.diag([drag_xy, drag_xy, drag_z])
```

!!! note
    `sys_id_translation` returns `drag_xy_coef` and `drag_z_coef` as scalars. The model expects a `(3, 3)` diagonal `drag_matrix` — assemble it as shown above.

## Which model to identify

Choose based on which physical effects you need to capture:

- **`so_rpy`** — identifies only `cmd_f_coef`; no motor dynamics, no drag. Fastest to calibrate, good for slow flight.
- **`so_rpy_rotor`** — adds `thrust_time_coef` to model motor spin-up delay. Better for agile maneuvers.
- **`so_rpy_rotor_drag`** — adds `drag_xy_coef` and `drag_z_coef`. Best accuracy at higher speeds where aerodynamic drag is significant.

---

That covers the core of the package. The final page documents the lower-level transform and rotation utilities — helpful when you need to work directly with motor forces, PWM values, or angular velocity representations.
