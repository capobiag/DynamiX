"""JAX backend: functional, jit-compiled, same equations as the NumPy reference backend.

Importing this package enables 64-bit floats in JAX (``jax_enable_x64``) so results match the
NumPy backend. State containers are NamedTuple PyTrees; nothing is mutated in place.
"""

import jax

jax.config.update("jax_enable_x64", True)

from dynamix.jax_backend.engine import (  # noqa: E402
    Engine,
    SimulationState,
    SystemParams,
    make_params,
    make_state,
    rollout,
    run,
    step,
    total_energy,
)

__all__ = [
    "Engine",
    "SimulationState",
    "SystemParams",
    "make_params",
    "make_state",
    "rollout",
    "run",
    "step",
    "total_energy",
]
