"""Unit quaternion utilities, [x, y, z, w] order (JAX); see ``dynamix.core.rotations``."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp

from dynamix.core import rotations

multiply = partial(rotations.multiply, jnp)
from_rotvec = partial(rotations.from_rotvec, jnp)
to_matrix = partial(rotations.to_matrix, jnp)
integrate = partial(rotations.integrate, jnp)
skew = partial(rotations.skew, jnp)
