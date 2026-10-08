"""A cube of balls falling onto a tilted plane, shown in 3D with PyVista.

Usage: python examples/ball_pile.py [--balls 125] [--seconds 1.5] [--tilt 0.3]
                                    [--backend numpy|jax]
                                    [--save pile.gif] [--no-show]
The plane is tilted by --tilt radians about y; without friction the balls slide down the slope.
The JAX timing excludes compilation.
Requires pyvista (pip install -e ".[viz3d]").
"""

import argparse
import time

import numpy as np

from dynamix.core import compile_scene
from dynamix.ecs import SimulationConfig
from dynamix.numpy_backend import Engine, total_energy
from dynamix.scenes import build_ball_pile

FPS = 30


def simulate(n_balls, radius, seconds, tilt, backend="numpy", dt=1e-3):
    """Run the pile; return the frames, energy, lowest gap to the plane, elapsed time and steps."""
    config = SimulationConfig(dt=dt, max_contacts=2 * n_balls)
    buf = compile_scene(build_ball_pile(n_balls, radius, config=config, tilt=tilt))
    stride = max(1, int(round(1.0 / (FPS * dt))))
    steps = int(round(seconds / dt)) // stride * stride
    initial = buf.q[:, :3].copy()
    if backend == "jax":
        import dynamix.jax_backend as jb

        engine = jb.Engine(buf)
        state0 = engine.state
        engine.rollout(steps, stride).q.block_until_ready()  # compile; not timed
        engine.state = state0
        start = time.perf_counter()
        recorded = engine.rollout(steps, stride)
        recorded.q.block_until_ready()
        elapsed = time.perf_counter() - start
        frames = np.concatenate((initial[None], np.asarray(recorded.q)[:, :, :3]))
        final_q, energy = np.asarray(engine.state.q), engine.total_energy()
    else:
        engine = Engine(buf)
        frames = [initial]
        start = time.perf_counter()
        for k in range(steps):
            engine.step()
            if (k + 1) % stride == 0:
                frames.append(buf.q[:, :3].copy())
        elapsed = time.perf_counter() - start
        frames, final_q, energy = np.array(frames), buf.q, total_energy(buf)
    depth = (final_q[:, :3] @ buf.plane_normal[0] - buf.plane_offset[0] - radius).min()
    return frames, energy, depth, elapsed, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balls", type=int, default=125)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--tilt", type=float, default=0.3, help="plane tilt about y, radians")
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    parser.add_argument("--save", default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    frames, energy, depth, elapsed, steps = simulate(
        args.balls, args.radius, args.seconds, args.tilt
    )
    print(
        f"[{args.backend}] simulated {steps} steps in {elapsed:.2f} s "
        f"({1e3 * elapsed / steps:.3f} ms/step)"
    )
    print(f"{args.balls} balls, final energy {energy:.2f}, lowest gap to the plane {depth:.2e}")

    import pyvista as pv

    normal = np.array([np.sin(args.tilt), 0.0, np.cos(args.tilt)])
    start = frames[0]
    centre = frames[..., :3].reshape(-1, 3).mean(axis=0)
    centre -= (centre @ normal) * normal  # point of the plane below the balls' mean position
    size = 2.0 * np.abs(frames[..., :3].reshape(-1, 3) - centre).max()
    plane = pv.Plane(center=centre, direction=normal, i_size=size, j_size=size)

    cloud = pv.PolyData(start.copy())
    cloud["height"] = start @ normal  # initial height above the plane, fixed colour per ball
    template = pv.Sphere(radius=args.radius, theta_resolution=24, phi_resolution=24)

    plotter = pv.Plotter(off_screen=args.no_show, window_size=(900, 700))
    plotter.add_mesh(plane, color="lightgray", opacity=0.8, show_edges=True)
    actor = plotter.add_mesh(
        cloud.glyph(geom=template, scale=False, orient=False),
        scalars="height",
        cmap="viridis",
        show_scalar_bar=False,
        smooth_shading=True,
    )
    plotter.add_axes()
    plotter.camera_position = "iso"

    def show_frame(k):
        nonlocal actor
        cloud.points = frames[k]
        plotter.remove_actor(actor, render=False)
        actor = plotter.add_mesh(
            cloud.glyph(geom=template, scale=False, orient=False),
            scalars="height",
            cmap="viridis",
            show_scalar_bar=False,
            smooth_shading=True,
            reset_camera=False,
        )
        plotter.add_text(f"t = {k / FPS:.2f} s", name="time", font_size=10)

    if args.save:
        plotter.open_gif(args.save, fps=FPS)
        for k in range(len(frames)):
            show_frame(k)
            plotter.write_frame()
    if not args.no_show:
        plotter.show(interactive_update=True, auto_close=False)
        while not plotter._closed:
            for k in range(len(frames)):
                if plotter._closed:
                    break
                show_frame(k)
                plotter.update()
                time.sleep(1.0 / FPS)
    plotter.close()


if __name__ == "__main__":
    main()
