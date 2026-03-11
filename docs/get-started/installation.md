# Installation

## Requirements

Python 3.10 or later. No GPU required; GPU acceleration is optional via JAX.

## Install from PyPI

You can install `drone-models` directly from PyPI using pip:

```bash
pip install drone-models
```

This brings in NumPy, SciPy, CasADi, and `array-api-compat` as core dependencies. The models are immediately usable with NumPy arrays.

For system identification — fitting model parameters from your own flight data — you need the `sysid` extra, which adds JAX (used for auto-differentiation during optimization) and matplotlib:

```bash
pip install "drone-models[sysid]"
```

## Developer install with pixi

For development work, we use [pixi](https://pixi.sh) as the environment manager. Pixi is a conda-based tool that locks the full dependency tree — including native libraries like CasADi — into a reproducible environment. It also defines named feature environments so you can activate exactly the dependencies you need for a given task.

Clone the repository and install all environments in one step:

```bash
git clone https://github.com/utiasDSL/drone-models.git
cd drone-models
pixi install
```

The available environments are:

| Environment | Activate with | Includes |
|---|---|---|
| `default` | `pixi shell` | Core package + ruff linter |
| `sysid` | `pixi shell -e sysid` | Core + JAX + matplotlib for system identification |
| `tests` | `pixi shell -e tests` | Core + JAX + pytest + array-api-strict for running the test suite |
| `docs` | `pixi shell -e docs` | Core + MkDocs + mkdocstrings for building the documentation |

Each environment has pixi tasks associated with it:

```bash
# Run the test suite
pixi run -e tests tests

# Serve the documentation locally (live reload)
pixi run -e docs docs-serve

# Build the documentation
pixi run -e docs docs-build
```

## SCIPY_ARRAY_API

`drone-models` requires SciPy to be loaded with its experimental Array API support enabled. The package sets the `SCIPY_ARRAY_API=1` environment variable automatically at import time — but only if SciPy has not been imported yet. If your environment or another library imports SciPy before `drone_models`, you will see a `RuntimeError`.

The fix is straightforward: make sure `drone_models` is imported first.

```python
import drone_models   # must precede any scipy import
import scipy
```

Or set the variable in your shell before starting Python:

```bash
export SCIPY_ARRAY_API=1
```

## Supported drone configurations

The following drone configurations ship with pre-fitted parameters, covering both the brushed Crazyflie 2.x series and the brushless Crazyflie 2.1:

| `drone_model` | Platform | Mass |
|---|---|---|
| `cf2x_L250` | Crazyflie 2.x — L250 props | 31.9 g |
| `cf2x_P250` | Crazyflie 2.x — P250 props | 31.8 g |
| `cf2x_T350` | Crazyflie 2.x — T350 props | 37.9 g |
| `cf21B_500` | Crazyflie 2.1 Brushless — 500 props | 43.4 g |

If your drone is not listed, you can fit parameters from your own flight data using the [system identification pipeline](../user-guide/system-identification.md), then inject the identified values into any model.

## Verify

```python
from drone_models import available_models
list(available_models)  # ['first_principles', 'so_rpy', 'so_rpy_rotor', 'so_rpy_rotor_drag']
```
