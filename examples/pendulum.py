"""Simulate an n-fold rigid-link pendulum, report energy drift and animate the motion.

Usage: python examples/pendulum.py [--links 3] [--seconds 10] [--dt 1e-3] [--angle 1.5]
                                   [--save pendulum.gif] [--no-show]
Requires matplotlib (pip install -e ".[viz]").
"""

import argparse

import numpy as np

from dynamix.core import compile_scene
from dynamix.ecs import SimulationConfig
from dynamix.numpy_backend import Engine, total_energy
from dynamix.numpy_backend import quaternion as quat
from dynamix.numpy_backend.constraints import ball_joint_residual
from dynamix.scenes import build_chain

FPS = 50


def simulate(links: int, angle: float, seconds: float, dt: float):
    """Run the simulation; return (times, points) per frame plus energy drift and max joint gap.

    ``points`` has shape (frames, links + 1, 3): the pivot, every joint and the free end.
    """
    scene = build_chain(links, angle, config=SimulationConfig(dt=dt))
    buf = compile_scene(scene)
    engine = Engine(buf)
    e0 = total_energy(buf)
    energy_scale = float(buf.mass.sum() * np.linalg.norm(buf.gravity) * links)

    # Joint b-anchors sit on each body's upper end; the last body's lower end is the free tip.
    upper = buf.joint_anchor_b[buf.joint_b.argsort()]
    tip_local = -upper[-1]

    def snapshot():
        rot = quat.to_matrix(buf.q[:, 3:])
        ends = buf.q[:, :3] + np.einsum("nij,nj->ni", rot, upper)
        tip = buf.q[-1, :3] + rot[-1] @ tip_local
        return np.vstack((ends, tip))

    stride = max(1, round(1.0 / (FPS * dt)))
    times, frames = [], []
    max_gap = 0.0
    for k in range(round(seconds / dt) + 1):
        if k % stride == 0:
            times.append(engine.time)
            frames.append(snapshot())
        engine.step()
        g = ball_joint_residual(
            buf.q, buf.joint_a, buf.joint_b, buf.joint_anchor_a, buf.joint_anchor_b
        )
        max_gap = max(max_gap, float(np.abs(g).max()))
    drift = (total_energy(buf) - e0) / energy_scale
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
    parser.add_argument("--save", help="write the animation to this file (e.g. pendulum.gif)")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args()

    times, points, drift, max_gap = simulate(args.links, args.angle, args.seconds, args.dt)
    print(f"energy drift / (m g L): {drift:.2e}, max joint gap: {max_gap:.1e}")
    animate(times, points, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
