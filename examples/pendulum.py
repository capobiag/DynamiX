"""Simulate a rigid-body pendulum, report energy drift and animate the motion.

Usage: python examples/pendulum.py [--seconds 5] [--dt 1e-3] [--angle 0.5]
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
from dynamix.scenes import build_pendulum

FPS = 50


def simulate(angle: float, seconds: float, dt: float):
    """Run the simulation; return (times, pivot, tip) per frame plus energy drift."""
    scene = build_pendulum(angle=angle, config=SimulationConfig(dt=dt))
    buf = compile_scene(scene)
    engine = Engine(buf)
    e0 = total_energy(buf)

    # Rod ends in the body frame: pivot at anchor_b, free end opposite to it.
    pivot_local = buf.joint_anchor_b[0]
    tip_local = -pivot_local

    def snapshot():
        rot = quat.to_matrix(buf.q[0, 3:])
        return buf.q[0, :3] + rot @ pivot_local, buf.q[0, :3] + rot @ tip_local

    stride = max(1, round(1.0 / (FPS * dt)))
    times, pivots, tips = [], [], []
    max_gap = 0.0
    for k in range(round(seconds / dt) + 1):
        if k % stride == 0:
            p, t = snapshot()
            times.append(engine.time)
            pivots.append(p)
            tips.append(t)
        engine.step()
        g = ball_joint_residual(
            buf.q, buf.joint_a, buf.joint_b, buf.joint_anchor_a, buf.joint_anchor_b
        )
        max_gap = max(max_gap, float(np.linalg.norm(g)))
    drift = (total_energy(buf) - e0) / abs(e0)
    return np.array(times), np.array(pivots), np.array(tips), drift, max_gap


def animate(times, pivots, tips, save=None, show=True):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    reach = 1.2 * np.max(np.linalg.norm(tips - pivots, axis=1))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(pivots[0, 0] - reach, pivots[0, 0] + reach)
    ax.set_ylim(pivots[0, 2] - reach, pivots[0, 2] + 0.2 * reach)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.grid(alpha=0.3)
    (rod,) = ax.plot([], [], "o-", lw=4, color="tab:blue", markersize=8)
    (trace,) = ax.plot([], [], lw=1, color="tab:orange", alpha=0.7)
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def update(i):
        rod.set_data([pivots[i, 0], tips[i, 0]], [pivots[i, 2], tips[i, 2]])
        trace.set_data(tips[: i + 1, 0], tips[: i + 1, 2])
        label.set_text(f"t = {times[i]:.2f} s")
        return rod, trace, label

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
    parser.add_argument("--angle", type=float, default=0.5, help="initial angle [rad]")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--save", help="write the animation to this file (e.g. pendulum.gif)")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args()

    times, pivots, tips, drift, max_gap = simulate(args.angle, args.seconds, args.dt)
    print(f"relative energy drift: {drift:.2e}, max joint gap: {max_gap:.1e}")
    animate(times, pivots, tips, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
