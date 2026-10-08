"""Unit quaternion utilities, [x, y, z, w] order (NumPy); see ``dynamix.core.rotations``."""

from __future__ import annotations

from functools import partial

import numpy as np

from dynamix.core import rotations

multiply = partial(rotations.multiply, np)
from_rotvec = partial(rotations.from_rotvec, np)
to_matrix = partial(rotations.to_matrix, np)
integrate = partial(rotations.integrate, np)
skew = partial(rotations.skew, np)
