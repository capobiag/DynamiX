"""Bouncing ball on a plane: height over time and an animation.

Usage: python examples/bouncing_ball.py [--height 1.0] [--restitution 0.8] [--seconds 3]
                                        [--backend numpy|jax] [--save ball.gif] [--no-show]
Requires matplotlib (pip install -e ".[viz]"); the jax backend also needs ".[jax]".
"""

import argparse

import numpy as np

from dynamix.core import compile_scene
from dynamix.ecs import SimulationConfig
from dynamix.scenes import build_bouncing_ball

FPS = 50


def simulate(height, radius, restitution, seconds, dt=1e-3, backend="numpy"):
    buf = compile_scene(
        build_bouncing_ball(height, radius, 1.0, restitution, SimulationConfig(dt=dt))
    )
    steps = int(round(seconds / dt))
    if backend == "jax":
        import dynamix.jax_backend as jb

        # the JAX engine keeps its own state; ``buf`` is not updated
        q = np.asarray(jb.Engine(buf).rollout(steps).q)
        z = np.concatenate(([buf.q[0, 2]], q[:, 0, 2]))
    else:
        from dynamix.numpy_backend import Engine

        engine = Engine(buf)
        z = np.empty(steps + 1)
        z[0] = buf.q[0, 2]
        for k in range(steps):
            engine.step()
            z[k + 1] = buf.q[0, 2]
    return np.arange(steps + 1) * dt, z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--restitution", type=float, default=0.8)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    parser.add_argument("--save", default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    t, z = simulate(args.height, args.radius, args.restitution, args.seconds, backend=args.backend)
    print(f"min centre height {z.min():.4f} (radius {args.radius})")

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    stride = max(1, int(round(1.0 / (FPS * (t[1] - t[0])))))
    frames = range(0, t.size, stride)
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.05, args.height + 0.2)
    ax.set_aspect("equal")
    ax.axhline(0.0, color="k")
    ball = plt.Circle((0.0, z[0]), args.radius, color="tab:blue")
    ax.add_patch(ball)

    def update(k):
        ball.center = (0.0, z[k])
        ax.set_title(f"t = {t[k]:.2f} s")
        return (ball,)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
    if args.save:
        anim.save(args.save, writer="pillow", fps=FPS)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
