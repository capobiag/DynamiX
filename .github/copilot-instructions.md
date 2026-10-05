# Copilot instructions

## Project

DynamiX is a modular, performance-focused multibody physics engine with
frictional contact. Preserve numerical correctness and follow the existing
codebase when implementing or changing behavior.

## Hardware and backends

- Local development runs on Apple Silicon macOS.
- For JAX locally, target CPU or MPS. Do not assume CUDA-enabled `jaxlib`.
- For Warp locally, target CPU. Keep Warp code compatible with CUDA on
  supported Linux or CI environments.
- Keep the JAX and Warp implementations separate. Do not mix their types or
  transfer arrays between backends inside solver loops.

## Architecture

- Use maximal coordinates `q`: each body has a Cartesian position, quaternion
  orientation, linear velocity, and angular velocity (six degrees of freedom).
- Use ECS only for topology, orchestration, and configuration.
- Before simulation, prepare solver inputs as flat, preallocated, contiguous
  numerical buffers. Do not access or iterate over ECS components in simulation
  loops or Warp kernels.
- Keep both backends aligned with the same engine interface while respecting
  their different execution models.

## Backend implementation

### JAX

- Keep `jax_solver/` code functional and stateless; use `jax.jit`, `jax.vmap`,
  and `jax.grad` where appropriate.
- Use `jax.numpy` rather than standard NumPy in solver code.
- Do not mutate arrays in place.
- Represent state containers as `NamedTuple` PyTrees.

### Warp

- Use Warp kernels (`@wp.kernel`) and imperative, parallel execution.
- Use flat, batched arrays with `ndim=2` where appropriate.
- Avoid dynamic or variable-length allocations in solver code.

## Physics conventions

- Use a right-handed, Z-up coordinate system.
- Store unit quaternions in `[x, y, z, w]` order.
- Express bilateral constraints as `g(t, q) = 0`.

## Working in the repository

- Inspect the relevant code and tests before changing behavior. Follow
  established APIs, naming, and formatting; do not assume files or tools that
  are not present.
- Keep changes focused. Update directly related documentation and tests when
  behavior or public interfaces change.
- Run the narrowest relevant validation available and report any checks that
  could not be run.
