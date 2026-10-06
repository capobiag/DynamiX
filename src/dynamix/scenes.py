"""Ready-made scenes."""

from __future__ import annotations

import numpy as np

from dynamix.ecs import WORLD, BallJoint, Body, Gravity, Scene, SimulationConfig


def build_pendulum(
    angle: float = 0.5,
    length: float = 1.0,
    mass: float = 1.0,
    config: SimulationConfig | None = None,
    gravity: float = 9.81,
) -> Scene:
    """Slender rod hinged at the world origin by a ball joint, in the x-z plane.

    ``angle`` is the rotation about +y from the downward vertical; the rod spans ``length``
    from the pivot, with its center of mass at ``length / 2``.
    """
    half = 0.5 * length
    inertia_t = mass * length**2 / 12.0
    inertia_axial = 1e-3 * inertia_t
    scene = Scene(config, Gravity(np.array([0.0, 0.0, -gravity])))
    body = scene.add_body(
        Body(
            mass=mass,
            inertia=np.array([inertia_t, inertia_t, inertia_axial]),
            position=np.array([-half * np.sin(angle), 0.0, -half * np.cos(angle)]),
            orientation=np.array([0.0, np.sin(0.5 * angle), 0.0, np.cos(0.5 * angle)]),
        )
    )
    scene.add_ball_joint(
        BallJoint(WORLD, body, anchor_a=np.zeros(3), anchor_b=np.array([0.0, 0.0, half]))
    )
    return scene
