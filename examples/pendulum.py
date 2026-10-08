"""Simulate an n-fold rigid-link pendulum, report energy drift and animate the motion.

Usage: python examples/pendulum.py [--links 3] [--seconds 10] [--dt 1e-3] [--angle 1.5]
                                   [--backend numpy|jax] [--save pendulum.gif] [--no-show]
Requires matplotlib (pip install -e ".[viz]"); the jax backend also needs ".[jax]".
"""

import argparse

import numpy as np

from dynamix.core import compile_scene
from dynamix.ecs import SimulationConfig
from dynamix.numpy_backend import quaternion as quat
from dynamix.numpy_backend.constraints import ball_joint_residual
from dynamix.scenes import build_chain

FPS = 50


class Simulation:
    """Uniform access to the NumPy and JAX engines: advance, positions, orientations, energy."""

    def __init__(self, backend: str, buf):
        self.backend = backend
        if backend == "jax":
            import dynamix.jax_backend as jb

            self.engine = jb.Engine(buf)
        else:
            from dynamix.numpy_backend import Engine

            self.engine = Engine(buf)
        self.buf = buf

    @property
    def time(self) -> float:
        return self.engine.time

    @property
    def q(self) -> np.ndarray:
        return np.asarray(self.engine.state.q if self.backend == "jax" else self.buf.q)

    def run(self, steps: int) -> None:
        self.engine.run(steps)

    def energy(self) -> float:
        if self.backend == "jax":
            return self.engine.total_energy()
        from dynamix.numpy_backend import total_energy

        return total_energy(self.buf)


def simulate(links: int, angle: float, seconds: float, dt: float, backend: str = "numpy"):
    """Run the simulation; return (times, points) per frame plus energy drift and max joint gap.

    ``points`` has shape (frames, links + 1, 3): the pivot, every joint and the free end. The
    joint gap is sampled once per frame.
    """
    scene = build_chain(links, angle, config=SimulationConfig(dt=dt))
    buf = compile_scene(scene)
    sim = Simulation(backend, buf)
    e0 = sim.energy()
    energy_scale = float(buf.mass.sum() * np.linalg.norm(buf.gravity) * links)

    # Joint b-anchors sit on each body's upper end; the last body's lower end is the free tip.
    upper = buf.joint_anchor_b[buf.joint_b.argsort()]
    tip_local = -upper[-1]

    def snapshot(q):
        rot = quat.to_matrix(q[:, 3:])
        ends = q[:, :3] + np.einsum("nij,nj->ni", rot, upper)
        tip = q[-1, :3] + rot[-1] @ tip_local
        return np.vstack((ends, tip))

    stride = max(1, round(1.0 / (FPS * dt)))
    times, frames = [], []
    max_gap = 0.0
    for _ in range(round(seconds / dt) // stride + 1):
        q = sim.q.copy()
        times.append(sim.time)
        frames.append(snapshot(q))
        g = ball_joint_residual(q, buf.joint_a, buf.joint_b, buf.joint_anchor_a, buf.joint_anchor_b)
        max_gap = max(max_gap, float(np.abs(g).max()))
        sim.run(stride)
    drift = (sim.energy() - e0) / energy_scale
    return np.array(times), np.array(frames), drift, max_gap


def animate(times, points, save=None, show=True):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    reach = 1.1 * np.max(np.linalg.norm(np.diff(points, axis=1), axis=2).sum(axis=1))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(points[0, 0, 0] - reach, points[0, 0, 0] + reach)
    ax.set_ylim(points[0, 0, 2] - reach, points[0, 0, 2] + 0.2 * reach)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.grid(alpha=0.3)
    (rods,) = ax.plot([], [], "o-", lw=4, color="tab:blue", markersize=8)
    (trace,) = ax.plot([], [], lw=1, color="tab:orange", alpha=0.7)
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    tips = points[:, -1]

    def update(i):
        rods.set_data(points[i, :, 0], points[i, :, 2])
        trace.set_data(tips[: i + 1, 0], tips[: i + 1, 2])
        label.set_text(f"t = {times[i]:.2f} s")
        return rods, trace, label

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / FPS, blit=True)
    if save:
        # Pillow writes GIFs without requiring ffmpeg
        writer = "pillow" if save.lower().endswith(".gif") else None
        anim.save(save, fps=FPS, writer=writer)
        print(f"saved animation to {save}")
    if show:
        plt.show()
    plt.close(fig)
    return anim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--links", type=int, default=3, help="number of rigid links")
    parser.add_argument(
        "--angle", type=float, default=1.5, help="initial angle of every link from vertical [rad]"
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--save", help="write the animation to this file (e.g. pendulum.gif)")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args()

    times, points, drift, max_gap = simulate(
        args.links, args.angle, args.seconds, args.dt, args.backend
    )
    print(f"energy drift / (m g L): {drift:.2e}, max joint gap: {max_gap:.1e}")
    animate(times, points, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
