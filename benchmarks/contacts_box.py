"""Contact-heavy NumPy vs JAX benchmark: a cubic stack of balls resting in a closed box.

Usage: python benchmarks/contacts_box.py [--sides 10 20 30] [--capacity 1 2 4]
                                         [--iterations 20] [--converged] [--steps 20]
                                         [--slots 8] [--float32] [--no-jax]
The scene (``build_ball_box``) has a known, constant contact set: ``3 s^2 (s - 1)`` sphere pairs
plus ``5 s^2`` plane contacts for ``s^3`` balls, so every timed step does the same work.

Two modes:
- fixed work (default): ``contact_tolerance = 0``, so both backends run exactly ``--iterations``
  sweeps per step. ``sweep`` is the cost of one sweep per contact, in ns.
- ``--converged``: the default tolerance with an iteration cap of ``--iterations``; reports the
  sweeps NumPy needed in the last step and the largest overlap, to compare accuracy as well.

For each size, NumPy runs once (its cost follows the active contacts) and JAX once per buffer
size ``--capacity`` x contacts (its cost follows the buffer). ``detect`` is the detection alone at
the same configuration, ``solve`` the rest of the step. Times are the best of ``--repeats``
chunks of ``--steps`` steps, after a warm-up chunk (which includes JAX compilation).
The lattice spacing is just below the JAX grid cell (a diameter), so a cell holds up to 8 balls;
``--slots`` must be at least that or JAX reports ``overflow`` and misses contacts.
"""

import argparse
import time

import numpy as np

from dynamix.core import compile_scene
from dynamix.ecs import SimulationConfig
from dynamix.numpy_backend import Engine
from dynamix.numpy_backend.contacts import ContactDetector
from dynamix.scenes import build_ball_box


def contact_count(side: int) -> int:
    return 3 * side**2 * (side - 1) + 5 * side**2


def best_ms(advance, steps: int, repeats: int, sync=lambda: None) -> float:
    """Best time per step in ms over ``repeats`` chunks of ``steps`` steps."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        advance(steps)
        sync()
        best = min(best, (time.perf_counter() - start) / steps * 1e3)
    return best


def make_buffers(side: int, capacity: int, args):
    config = SimulationConfig(
        max_contacts=capacity,
        contact_iterations=args.iterations,
        contact_tolerance=1e-9 if args.converged else 0.0,
    )
    return compile_scene(build_ball_box(side, config=config))


def run_numpy(side: int, args) -> dict:
    count = contact_count(side)
    buf = make_buffers(side, count, args)
    engine = Engine(buf)
    engine.run(args.steps)
    step = best_ms(engine.run, args.steps, args.repeats)
    detector = ContactDetector(buf)
    detect = best_ms(lambda k: [detector.detect(buf.q) for _ in range(k)], args.steps, args.repeats)
    contacts = detector.detect(buf.q)  # at the final configuration, as for JAX
    return {
        "contacts": contacts.count,
        "buffer": count,
        "step": step,
        "detect": detect,
        "iterations": engine._contact_solver.last_iterations,
        "overlap": float(-contacts.gap[: contacts.count].min()),
        "overflow": contacts.overflow,
    }


def run_jax(side: int, factor: float, args) -> dict:
    import jax
    import jax.numpy as jnp

    import dynamix.jax_backend as jb
    from dynamix.jax_backend import contacts as bc

    capacity = int(np.ceil(factor * contact_count(side)))
    buf = make_buffers(side, capacity, args)
    engine = jb.Engine(buf, contact_slots=args.slots)
    if args.float32:

        def single(x):
            return x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x

        engine.state = jax.tree_util.tree_map(single, engine.state)
        engine.params = jax.tree_util.tree_map(single, engine.params)

    def sync():
        engine.state.q.block_until_ready()

    engine.run(args.steps)  # includes compilation
    sync()
    step = best_ms(engine.run, args.steps, args.repeats, sync)

    detect_fn = jax.jit(bc.detect, static_argnums=1)
    cp, cfg, q = engine.params.contacts, engine.contact_cfg, engine.state.q
    out = {}

    def detect(k):
        for _ in range(k):
            out["c"] = detect_fn(cp, cfg, q)
        out["c"].gap.block_until_ready()

    detect(1)
    detect_ms = best_ms(detect, args.steps, args.repeats)
    contacts = out["c"]
    kept = int(contacts.count)
    return {
        "contacts": kept,
        "buffer": capacity,
        "step": step,
        "detect": detect_ms,
        "iterations": None,
        "overlap": float(-np.asarray(contacts.gap)[:kept].min()),
        "overflow": bool(contacts.overflow),
        "dtype": str(engine.state.q.dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sides", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument(
        "--capacity", type=float, nargs="+", default=[1.0, 2.0, 4.0], help="buffer / contacts"
    )
    parser.add_argument("--iterations", type=int, default=20, help="sweeps (cap) per step")
    parser.add_argument("--converged", action="store_true", help="stop at the tolerance")
    parser.add_argument("--steps", type=int, default=20, help="steps per timed chunk")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--slots", type=int, default=8, help="JAX spheres scanned per cell")
    parser.add_argument("--float32", action="store_true", help="run JAX in single precision")
    parser.add_argument("--no-jax", action="store_true")
    args = parser.parse_args()

    mode = "converged" if args.converged else "fixed work"
    print(f"mode: {mode}, {args.iterations} sweeps{' (cap)' if args.converged else ''} per step")
    print(
        f"{'balls':>6} {'contacts':>8} {'backend':>8} {'buffer':>7} {'ms/step':>8} "
        f"{'detect':>7} {'solve':>7} {'sweep ns':>8} {'iters':>5} {'overlap':>9} {'speedup':>7}"
    )
    for side in args.sides:
        runs = [("numpy", 1.0, run_numpy(side, args))]
        if not args.no_jax:
            name = "jax32" if args.float32 else "jax"
            runs += [(name, f, run_jax(side, f, args)) for f in args.capacity]
        reference = runs[0][2]["step"]
        for name, _, r in runs:
            solve = r["step"] - r["detect"]
            # per sweep and contact; only meaningful when the sweep count is fixed
            sweep = (
                "-" if args.converged else f"{solve / args.iterations / r['contacts'] * 1e6:.1f}"
            )
            iters = "-" if r["iterations"] is None or not args.converged else r["iterations"]
            note = " overflow" if r["overflow"] else ""
            if r.get("dtype") not in (None, "float64", "float32"):
                note += f" ({r['dtype']})"
            print(
                f"{side**3:>6} {r['contacts']:>8} {name:>8} {r['buffer']:>7} {r['step']:>8.3f} "
                f"{r['detect']:>7.3f} {solve:>7.3f} {sweep:>8} {iters!s:>5} "
                f"{r['overlap']:>9.2e} {reference / r['step']:>6.1f}x{note}"
            )


if __name__ == "__main__":
    main()
