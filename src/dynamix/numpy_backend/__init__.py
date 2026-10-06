"""Python/NumPy reference backend (CPU only; no JAX or Warp required)."""

from dynamix.numpy_backend.engine import Engine, total_energy

__all__ = ["Engine", "total_energy"]
