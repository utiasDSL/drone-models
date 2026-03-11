"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

SKIP_PARTS = {"_typing", "__main__"}

# model.py in these packages is re-exported via __init__, so document only at index level
SKIP_MODULE_PAGES = {
    ("drone_models", "first_principles", "model"),
    ("drone_models", "so_rpy", "model"),
    ("drone_models", "so_rpy_rotor", "model"),
    ("drone_models", "so_rpy_rotor_drag", "model"),
}

TOP_LEVEL_PAGE = """\
::: drone_models
    options:
      members:
        - parametrize
        - available_models
        - model_features
      show_submodules: false

## Submodules

### Models

| Module | Description |
|--------|-------------|
| [first_principles](first_principles/index.md) | Full physics model |
| [so_rpy_rotor_drag](so_rpy_rotor_drag/index.md) | Fitted model with rotor dynamics and drag |
| [so_rpy_rotor](so_rpy_rotor/index.md) | Fitted model with rotor dynamics |
| [so_rpy](so_rpy/index.md) | Simplest fitted model |

### Core

| Module | Description |
|--------|-------------|
| [core](../core.md) | `load_params`, `supports` decorator, internal `parametrize` |
| [drones](../drones.md) | `available_drones` tuple |
| [symbols](../symbols.md) | CasADi symbolic utilities |
| [transform](../transform.md) | Motor/PWM/rotor conversion utilities |

### Utilities

| Module | Description |
|--------|-------------|
| [utils.rotation](../utils/rotation.md) | Quaternion/Euler/angular velocity conversions |
| [utils.data_utils](../utils/data_utils.md) | `preprocessing`, `derivatives_svf` |
| [utils.identification](../utils/identification.md) | `sys_id_translation`, `sys_id_rotation` |
"""

for path in sorted(Path("drone_models").rglob("*.py")):
    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to(".").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if any(part in SKIP_PARTS for part in parts):
        continue

    if parts in SKIP_MODULE_PAGES:
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        if parts == ("drone_models",):
            fd.write(TOP_LEVEL_PAGE)
        else:
            ident = ".".join(parts)
            fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

summary = """\
* [drone_models](drone_models/index.md)
* [first_principles](drone_models/first_principles/index.md)
* [so_rpy_rotor_drag](drone_models/so_rpy_rotor_drag/index.md)
* [so_rpy_rotor](drone_models/so_rpy_rotor/index.md)
* [so_rpy](drone_models/so_rpy/index.md)
* Core
    * [core](drone_models/core.md)
    * [drones](drone_models/drones.md)
    * [symbols](drone_models/symbols.md)
    * [transform](drone_models/transform.md)
* Utilities
    * [utils](drone_models/utils/index.md)
    * [utils.rotation](drone_models/utils/rotation.md)
    * [utils.data_utils](drone_models/utils/data_utils.md)
    * [utils.identification](drone_models/utils/identification.md)
"""

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.write(summary)
