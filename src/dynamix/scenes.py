"""Ready-made scenes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dynamix.ecs import WORLD, BallJoint, Body, Gravity, HingeJoint, Scene, SimulationConfig


def _per_link(value: float | Sequence[float], n: int, name: str) -> np.ndarray:
    arr = np.broadcast_to(np.asarray(value, dtype=np.float64), (n,)).copy()
    if arr.shape != (n,) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive scalars or a sequence of length {n}")
    return arr


def build_chain(
    n_links: int = 2,
    angles: float | Sequence[float] = np.pi / 2,
    length: float | Sequence[float] = 1.0,
    mass: float | Sequence[float] = 1.0,
    config: SimulationConfig | None = None,
    gravity: float = 9.81,
    joint: str = "ball",
) -> Scene:
    """Chain of ``n_links`` slender rods (an n-fold pendulum) hung from the world origin.

    Rods lie in the x-z plane, joined end to end (and to the world) by ball joints. Each body's
    local z axis runs along its rod with the upper joint at ``+length / 2``. ``angles[i]`` is the
    absolute rotation of link i about +y from the downward vertical (default: horizontal, at rest).
    ``joint`` is ``"ball"`` or ``"hinge"`` (hinges about the y axis, i.e. a planar chain).
    """
    if joint not in ("ball", "hinge"):
        raise ValueError("joint must be 'ball' or 'hinge'")
    if n_links < 1:
        raise ValueError("n_links must be at least 1")
    phi = np.broadcast_to(np.asarray(angles, dtype=np.float64), (n_links,))
    lengths = _per_link(length, n_links, "length")
    masses = _per_link(mass, n_links, "mass")

    scene = Scene(config, Gravity(np.array([0.0, 0.0, -gravity])))
    pivot = np.zeros(3)
    previous = WORLD
    previous_anchor = np.zeros(3)
    for i in range(n_links):
        half = 0.5 * lengths[i]
        down = np.array([-np.sin(phi[i]), 0.0, -np.cos(phi[i])])
        inertia_t = masses[i] * lengths[i] ** 2 / 12.0
        body = scene.add_body(
            Body(
                mass=masses[i],
                inertia=np.array([inertia_t, inertia_t, 1e-3 * inertia_t]),
                position=pivot + half * down,
                orientation=np.array([0.0, np.sin(0.5 * phi[i]), 0.0, np.cos(0.5 * phi[i])]),
            )
        )
        anchor_b = np.array([0.0, 0.0, half])
        if joint == "ball":
            scene.add_ball_joint(BallJoint(previous, body, previous_anchor, anchor_b))
        else:
            axis = np.array([0.0, 1.0, 0.0])
            scene.add_hinge_joint(HingeJoint(previous, body, previous_anchor, anchor_b, axis, axis))
        pivot = pivot + lengths[i] * down
        previous, previous_anchor = body, np.array([0.0, 0.0, -half])
    return scene


def build_pendulum(
    angle: float = 0.5,
    length: float = 1.0,
    mass: float = 1.0,
    config: SimulationConfig | None = None,
    gravity: float = 9.81,
    joint: str = "ball",
) -> Scene:
    """Single slender rod hung from the world origin by a ball or hinge joint, in the x-z plane."""
    return build_chain(1, angle, length, mass, config, gravity, joint)
