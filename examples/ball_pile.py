"""A cube of balls falling onto a tilted plane, animated (NumPy backend).

Usage: python examples/ball_pile.py [--balls 125] [--seconds 1.5] [--tilt 0.3]
                                    [--save pile.gif] [--no-show]
The plane is tilted by --tilt radians about y; without friction the balls slide down the slope.
Requires matplotlib (pip install -e ".[viz]").
"""

import argparse

import numpy as np

from dynamix.core import compile_scene
from dynamix.ecs import SimulationConfig
from dynamix.numpy_backend import Engine, total_energy
from dynamix.scenes import build_ball_pile

FPS = 30


def simulate(n_balls, radius, seconds, tilt, dt=1e-3):
    config = SimulationConfig(dt=dt, max_contacts=12 * n_balls)
    buf = compile_scene(build_ball_pile(n_balls, radius, config=config, tilt=tilt))
    engine = Engine(buf)
    stride = max(1, int(round(1.0 / (FPS * dt))))
    frames = [buf.q[:, :3].copy()]
    for k in range(int(round(seconds / dt))):
        engine.step()
        if (k + 1) % stride == 0:
            frames.append(buf.q[:, :3].copy())
    normal = buf.plane_normal[0]
    depth = (buf.q[:, :3] @ normal - buf.plane_offset[0] - radius).min()
    return np.array(frames), total_energy(buf), depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balls", type=int, default=125)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--tilt", type=float, default=0.3, help="plane tilt about y, radians")
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--save", default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    frames, energy, depth = simulate(args.balls, args.radius, args.seconds, args.tilt)
    print(f"{args.balls} balls, final energy {energy:.2f}, lowest gap to the plane {depth:.2e}")

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    pad = 2 * args.radius
    x_lo, x_hi = frames[..., 0].min() - pad, frames[..., 0].max() + pad
    z_lo, z_hi = frames[..., 2].min() - pad, frames[..., 2].max() + pad
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_lo, z_hi)
    ax.set_aspect("equal")
    normal = np.array([np.sin(args.tilt), np.cos(args.tilt)])  # (x, z) of the plane normal
    xs = np.array([x_lo, x_hi])
    ax.plot(xs, -normal[0] / normal[1] * xs, color="k")
    # side view (x-z); marker size in points^2 for a diameter of 2 * radius in data units
    points_per_unit = fig.get_figwidth() * 72 / (x_hi - x_lo) * 0.8
    scatter = ax.scatter(
        frames[0][:, 0], frames[0][:, 2], s=(2 * args.radius * points_per_unit) ** 2, alpha=0.6
    )

    def update(k):
        scatter.set_offsets(frames[k][:, [0, 2]])
        ax.set_title(f"frame {k}")
        return (scatter,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, blit=False)
    if args.save:
        anim.save(args.save, writer="pillow", fps=FPS)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
