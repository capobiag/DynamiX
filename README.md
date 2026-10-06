# DynamiX
Ultra-fast, modular multibody systems physics engine for systems with frictional contact.

## Getting started

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pip install -e ".[viz]"        # matplotlib, for the pendulum animation
.venv/bin/python examples/pendulum.py    # n-fold pendulum (default 3 links); opens an animation window
.venv/bin/python examples/pendulum.py --links 5 --save pendulum.gif --no-show   # write a GIF instead
.venv/bin/pytest
```

Current scope: a NumPy/CPU backend simulating rigid bodies in maximal coordinates
(position, `[x, y, z, w]` quaternion, 6 DOF; Z-up; linear velocity in the world frame, angular
velocity in the body frame) connected by ball joints, stepped with
Moreau's theta method and velocity-level bilateral constraints. Scenes are described with
[esper](https://github.com/benmoran56/esper) components (`dynamix.ecs`), compiled once into flat
NumPy buffers (`dynamix.core.compile_scene`), and stepped by `dynamix.numpy_backend.Engine`
without touching the ECS. `dynamix.scenes.build_chain(n)` builds an n-fold rigid-link pendulum;
it is validated against an independent Lagrangian reference in `tests/`. The constraint solve is block-sparse and banded (linear in chain length); install the optional
`fast` extra (SciPy) for the banded Cholesky, otherwise a dense NumPy solve is used.
Contacts, JAX and Warp backends are not implemented yet.

The mathematics behind the engine is described in [docs/theory.md](docs/theory.md).
