"""Many balls falling onto a plane and piling up (NumPy backend).

Usage: python benchmarks/contacts_balls.py [--steps 1000] [--sizes 100 500 2000]
Reports ms/step, the contact count, solver iterations, the deepest penetration and the energy.
"""

import argparse
import time

import numpy as np

from dynamix.core import compile_scene
from dynamix.ecs import SimulationConfig
from dynamix.numpy_backend import Engine, total_energy
from dynamix.scenes import build_ball_pile


def run(n_balls: int, steps: int, radius: float = 0.1):
    config = SimulationConfig(dt=1e-3, max_contacts=12 * n_balls)
    buf = compile_scene(build_ball_pile(n_balls, radius, config=config))
    engine = Engine(buf)
    engine.run(20)  # warm-up
    start = time.perf_counter()
    iterations = 0
    for _ in range(steps):
        engine.step()
        iterations += engine._contact_solver.last_iterations if engine.contacts.count else 0
    ms = (time.perf_counter() - start) / steps * 1e3
    pen = max(0.0, radius - (buf.q[:, :3] @ buf.plane_normal[0] - buf.plane_offset[0]).min())
    return ms, engine.contacts.count, iterations / steps, pen, total_energy(buf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 2000])
    args = parser.parse_args()
    print(
        f"{'balls':>6} {'ms/step':>9} {'contacts':>9} {'iters':>6} "
        f"{'penetration':>12} {'energy':>9}"
    )
    for n in args.sizes:
        ms, contacts, iters, pen, energy = run(n, args.steps)
        print(f"{n:>6} {ms:>9.3f} {contacts:>9} {iters:>6.1f} {pen:>12.2e} {energy:>9.2f}")
    assert np.isfinite(energy)


if __name__ == "__main__":
    main()
