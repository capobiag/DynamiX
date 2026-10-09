"""Ready-made scenes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dynamix.ecs import (
    WORLD,
    BallJoint,
    Body,
    Gravity,
    HingeJoint,
    PlaneCollider,
    Scene,
    SimulationConfig,
    SphereCollider,
)


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


def _add_ball(scene: Scene, position, radius: float, mass: float, restitution: float) -> int:
    inertia = 0.4 * mass * radius**2
    body = scene.add_body(Body(mass=mass, inertia=np.full(3, inertia), position=position))
    scene.add_sphere_collider(body, SphereCollider(radius, restitution))
    return body


def build_bouncing_ball(
    height: float = 1.0,
    radius: float = 0.1,
    mass: float = 1.0,
    restitution: float = 0.8,
    config: SimulationConfig | None = None,
    gravity: float = 9.81,
) -> Scene:
    """Ball dropped from rest, its centre at ``height``, onto the plane z = 0.

    The ball and the plane both carry ``restitution``.
    """
    if height < radius:
        raise ValueError("height must be at least the radius")
    scene = Scene(config, Gravity(np.array([0.0, 0.0, -gravity])))
    _add_ball(scene, [0.0, 0.0, height], radius, mass, restitution)
    scene.add_plane(PlaneCollider(restitution=restitution))
    return scene


def build_ball_pile(
    n_balls: int = 100,
    radius: float = 0.1,
    mass: float = 1.0,
    restitution: float = 0.0,
    config: SimulationConfig | None = None,
    gravity: float = 9.81,
    tilt: float = 0.3,
    clearance: float = 0.5,
    gap: float = 0.05,
) -> Scene:
    """A cube of balls dropped onto a plane that is tilted against the cube's axes.

    The balls sit on a simple cubic lattice aligned with the world axes, ``side**3 >= n_balls``
    with ``side = ceil(n_balls ** (1/3))``, filled layer by layer from the bottom. The spacing is
    ``2 radius (1 + gap)``, so no two balls overlap. The plane passes through the origin with the
    normal ``(sin(tilt), 0, cos(tilt))`` (a rotation of ``tilt`` radians about the y axis); the
    cube is centred over the origin and lifted until its lowest ball is ``clearance * radius``
    above the plane. Without friction the balls slide down the slope.
    """
    if n_balls < 1:
        raise ValueError("n_balls must be at least 1")
    if not 0.0 <= tilt < 0.5 * np.pi:
        raise ValueError("tilt must be in [0, pi/2)")
    if gap < 0.0 or clearance < 0.0:
        raise ValueError("gap and clearance must be non-negative")
    side = int(np.ceil(round(n_balls ** (1.0 / 3.0), 9)))
    spacing = 2.0 * radius * (1.0 + gap)
    index = np.arange(n_balls)
    cell = np.stack([index % side, (index // side) % side, index // side**2], axis=1)
    centres = (cell - 0.5 * (side - 1) * np.array([1.0, 1.0, 0.0])) * spacing
    normal = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    lift = (clearance + 1.0) * radius - (centres @ normal).min()
    centres[:, 2] += lift / normal[2]

    scene = Scene(config, Gravity(np.array([0.0, 0.0, -gravity])))
    for centre in centres:
        _add_ball(scene, centre, radius, mass, restitution)
    scene.add_plane(PlaneCollider(normal=normal, restitution=restitution))
    return scene


def build_ball_box(
    side: int = 10,
    radius: float = 0.1,
    mass: float = 1.0,
    config: SimulationConfig | None = None,
    gravity: float = 9.81,
    overlap: float = 1e-3,
) -> Scene:
    """A ``side**3`` cubic stack of balls resting in a closed box (floor and four walls).

    Neighbouring balls, the bottom layer and the floor, and the outer balls and the walls all
    overlap by ``overlap`` times the radius per surface (the lattice spacing is
    ``2 radius (1 - overlap)``), so every lattice neighbour is in contact from the first step:
    ``3 side**2 (side - 1)`` sphere pairs plus ``5 side**2`` plane contacts. The stack is
    symmetric, so (without friction) it stays a stack and the contact set does not change; this
    makes it a benchmark with a known, constant number of contacts.
    """
    if side < 1:
        raise ValueError("side must be at least 1")
    if not 0.0 < overlap < 0.5:
        raise ValueError("overlap must be in (0, 0.5)")
    spacing = 2.0 * radius * (1.0 - overlap)
    inset = radius * (1.0 - overlap)  # distance of the outer centres from the walls
    half = 0.5 * (side - 1) * spacing + inset
    index = np.arange(side**3)
    cell = np.stack([index % side, (index // side) % side, index // side**2], axis=1)
    centres = cell * spacing + np.array([inset - half, inset - half, inset])

    scene = Scene(config, Gravity(np.array([0.0, 0.0, -gravity])))
    for centre in centres:
        _add_ball(scene, centre, radius, mass, 0.0)
    scene.add_plane(PlaneCollider())
    for axis in (0, 1):
        for sign in (1.0, -1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            scene.add_plane(PlaneCollider(normal=normal, offset=-half))
    return scene
