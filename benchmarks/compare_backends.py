"""Performance comparison of the NumPy and JAX backends on n-link pendulum chains.

Usage: python benchmarks/compare_backends.py [--links 3 10 50 200 1000] [--repeats 5]

Reported times are milliseconds per step (best of ``--repeats``, compilation excluded):
  numpy      Python loop over ``Engine.step``
  jax step   Python loop over the jitted single step (includes per-call dispatch overhead)
  jax run    one compiled ``lax.scan`` over all steps (no Python in the loop)
Compilation time of the jax run is reported separately.
"""

import argparse
import time

import dynamix.jax_backend as jb
from dynamix.core import compile_scene
from dynamix.ecs import SimulationConfig
from dynamix.numpy_backend import Engine as NumpyEngine
from dynamix.scenes import build_chain


def best_of(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def make_buffers(n):
    return compile_scene(build_chain(n, config=SimulationConfig(dt=1e-3)))


def bench(n, steps, repeats):
    np_engine = NumpyEngine(make_buffers(n))
    np_engine.run(5)
    t_numpy = best_of(lambda: np_engine.run(steps), repeats) / steps

    engine = jb.Engine(make_buffers(n))
    engine.step()
    engine.state.q.block_until_ready()

    def loop():
        for _ in range(steps):
            engine.step()
        engine.state.q.block_until_ready()

    t_step = best_of(loop, repeats) / steps

    engine = jb.Engine(make_buffers(n))
    start = time.perf_counter()
    engine.run(steps)
    engine.state.q.block_until_ready()
    compile_s = time.perf_counter() - start

    def scan():
        engine.run(steps)
        engine.state.q.block_until_ready()

    t_run = best_of(scan, repeats) / steps
    return t_numpy * 1e3, t_step * 1e3, t_run * 1e3, compile_s


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--links", type=int, nargs="+", default=[1, 3, 10, 50, 200, 1000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=0, help="steps per timing (0 = automatic)")
    args = parser.parse_args()

    header = (
        f"{'links':>6} {'numpy':>10} {'jax step':>10} {'jax run':>10} {'speedup':>9} {'compile':>9}"
    )
    print(header)
    print(f"{'':>6} {'ms/step':>10} {'ms/step':>10} {'ms/step':>10} {'run/np':>9} {'s':>9}")
    for n in args.links:
        steps = args.steps or max(20, min(2000, int(2e5 / (n + 20))))
        t_np, t_step, t_run, compile_s = bench(n, steps, args.repeats)
        speedup = t_np / t_run
        print(f"{n:6d} {t_np:10.4f} {t_step:10.4f} {t_run:10.4f} {speedup:8.1f}x {compile_s:9.2f}")


if __name__ == "__main__":
    main()
