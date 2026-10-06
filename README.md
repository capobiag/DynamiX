# DynamiX
Ultra-fast, modular multibody systems physics engine for systems with frictional contact.

## Project structure

- `src/dynamix/` — engine source
  - `core/` — shared simulation interfaces and numerical utilities
  - `ecs/` — topology, orchestration, and configuration
  - `contact/` — contact data and detection
  - `numpy_solver/` — CPU reference backend
  - `jax_solver/` — JAX backend
  - `warp_solver/` — Warp backend
- `tests/` — unit and integration tests
- `examples/` — runnable examples
- `docs/` — project documentation
