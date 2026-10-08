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
velocity in the body frame) connected by ball, hinge, fixed or prismatic joints (`Scene.add_ball_joint`, `add_hinge_joint`, `add_fixed_joint`, `add_prismatic_joint`; `build_chain(..., joint="hinge")`), stepped with
Moreau's theta method and velocity-level bilateral constraints. Frictionless normal contacts
(sphere-sphere and sphere-plane via SDFs, Newton restitution; NumPy only so far) are added with
`SphereCollider`/`PlaneCollider`; see `build_bouncing_ball`, `build_ball_pile` and
`benchmarks/contacts_balls.py`; examples: `examples/bouncing_ball.py`, `examples/ball_pile.py`. Scenes are described with
[esper](https://github.com/benmoran56/esper) components (`dynamix.ecs`), compiled once into flat
NumPy buffers (`dynamix.core.compile_scene`), and stepped by `dynamix.numpy_backend.Engine`
without touching the ECS. `dynamix.scenes.build_chain(n)` builds an n-fold rigid-link pendulum;
it is validated against an independent Lagrangian reference in `tests/`. The constraint solve is block-sparse and banded (linear in chain length); install the optional
`fast` extra (SciPy) for the banded Cholesky, otherwise a dense NumPy solve is used.

`dynamix.jax_backend` (install the `jax` extra) implements the same engine functionally: immutable
`NamedTuple` state, a jitted `step`, a `lax.scan`-based `run`/`rollout`, and a block-banded
Cholesky for the joint solve; it is `vmap`/`grad`-compatible and agrees with the NumPy backend to
round-off. `benchmarks/compare_backends.py` compares the two (Apple M1, CPU, float64, ms/step):

| links | NumPy | JAX `step` loop | JAX `run` (scan) | speedup (scan/NumPy) |
|------:|------:|----------------:|-----------------:|---------------------:|
| 1     | 0.127 | 0.016           | 0.001            | 132x                 |
| 10    | 0.144 | 0.027           | 0.011            | 14x                  |
| 50    | 0.199 | 0.083           | 0.047            | 4.2x                 |
| 200   | 0.412 | 0.201           | 0.172            | 2.4x                 |
| 1000  | 1.532 | 0.864           | 0.872            | 1.8x                 |

The backend-independent math is written once in `dynamix.core` (`rotations`, `joints`, `joint_math`,
`joint_topology`) with the array namespace `xp` (`numpy` or `jax.numpy`) passed in and no
in-place operations; each backend keeps only its assembly, solver and time stepper.

Contacts and the Warp backend are not implemented yet.

The mathematics behind the engine is described in [docs/theory.md](docs/theory.md).
