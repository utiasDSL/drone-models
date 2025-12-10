from typing import Any, TypeAlias

import numpy.typing as npt

Array: TypeAlias = Any  # To be changed to a Protocol later (see array-api#589)
ArrayLike: TypeAlias = Array | npt.ArrayLike
