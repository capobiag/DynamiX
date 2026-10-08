"""Many balls falling onto a tilted plane: NumPy vs JAX contact performance.

Usage: python benchmarks/contacts_balls.py [--steps 500] [--sizes 125 1000 4000]
                                           [--capacity 2.0] [--no-jax]
Both backends simulate the same scene (the ball pile of ``build_ball_pile``). They run a warm-up
of ``--steps`` steps first, so the timings are for a pile that has hit the plane. The time is the
best of five chunks. The contact buffer holds ``capacity * n`` contacts: NumPy cost follows the
actual contact count, JAX cost follows the buffer size (static shapes), so keep it tight.
"""

import argparse
import time

import numpy as np

from dynamix.core import compile_scene
from dynamix.ecs import SimulationConfig
from dynamix.numpy_backend import Engine, total_energy
from dynamix.scenes import build_ball_pile

RADIUS = 0.1


def best_ms(advance, steps: int, sync=lambda: None) -> float:
    chunk = max(1, steps // 5)
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        advance(chunk)
        sync()
        best = min(best, (time.perf_counter() - start) / chunk * 1e3)
    return best


def penetration(q, buf) -> float:
    return max(0.0, RADIUS - float((q[:, :3] @ buf.plane_normal[0] - buf.plane_offset[0]).min()))


def run_numpy(n_balls: int, steps: int, capacity: float):
    config = SimulationConfig(max_contacts=int(capacity * n_balls))
    buf = compile_scene(build_ball_pile(n_balls, RADIUS, config=config))
    engine = Engine(buf)
    engine.run(steps)
    ms = best_ms(engine.run, steps)
    return ms, engine.contacts.count, penetration(buf.q, buf), total_energy(buf)


def run_jax(n_balls: int, steps: int, capacity: float):
    import dynamix.jax_backend as jb

    config = SimulationConfig(max_contacts=int(capacity * n_balls))
    buf = compile_scene(build_ball_pile(n_balls, RADIUS, config=config))
    engine = jb.Engine(buf)
    engine.run(steps)  # includes compilation
    ms = best_ms(engine.run, steps, lambda: engine.state.q.block_until_ready())
    contacts = engine.contacts
    if bool(contacts.overflow):
        print("  warning: JAX contact buffer overflow, raise --capacity")
    pen = penetration(np.asarray(engine.state.q), buf)
    return ms, int(contacts.count), pen, engine.total_energy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--sizes", type=int, nargs="+", default=[125, 1000, 4000])
    parser.add_argument("--capacity", type=float, default=2.0, help="contact buffer size per ball")
    parser.add_argument("--no-jax", action="store_true")
    args = parser.parse_args()
    print(
        f"{'balls':>6} {'backend':>8} {'ms/step':>9} {'contacts':>9} "
        f"{'penetration':>12} {'energy':>9}"
    )
    for n in args.sizes:
        runs = [("numpy", run_numpy)] + ([] if args.no_jax else [("jax", run_jax)])
        times = {}
        for name, fn in runs:
            ms, contacts, pen, energy = fn(n, args.steps, args.capacity)
            times[name] = ms
            print(f"{n:>6} {name:>8} {ms:>9.3f} {contacts:>9} {pen:>12.2e} {energy:>9.2f}")
        if "jax" in times:
            print(f"{'':>6} {'speedup':>8} {times['numpy'] / times['jax']:>8.1f}x")


if __name__ == "__main__":
    main()
