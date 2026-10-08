"""esper components. Plain data; never accessed inside simulation loops."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vec(values, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(n)
    return arr.copy()


@dataclass
class Body:
    """Rigid body in maximal coordinates (position, [x, y, z, w] quaternion, 6 DOF).

    Position is the center of mass; ``inertia`` is expressed in the body frame
    about the center of mass. Linear velocity is expressed in the world frame, angular
    velocity in the body frame.
    """

    mass: float
    inertia: np.ndarray
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        inertia = np.asarray(self.inertia, dtype=np.float64)
        self.inertia = np.diag(inertia) if inertia.ndim == 1 else inertia.reshape(3, 3).copy()
        self.position = _vec(self.position, 3)
        self.orientation = _vec(self.orientation, 4)
        self.linear_velocity = _vec(self.linear_velocity, 3)
        self.angular_velocity = _vec(self.angular_velocity, 3)
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")


@dataclass
class BallJoint:
    """Spherical joint: anchor_a (on body_a) coincides with anchor_b (on body_b).

    ``body_a`` is an entity id or ``WORLD``. Anchors are expressed in the body frame;
    for ``WORLD`` the anchor is a fixed world-frame point.
    """

    body_a: int
    body_b: int
    anchor_a: np.ndarray
    anchor_b: np.ndarray

    def __post_init__(self) -> None:
        self.anchor_a = _vec(self.anchor_a, 3)
        self.anchor_b = _vec(self.anchor_b, 3)


@dataclass
class HingeJoint:
    """Revolute joint: the anchors coincide and the hinge axes stay parallel.

    ``axis_a`` and ``axis_b`` are the hinge axis in the frame of each body (the world frame for
    ``WORLD``); they must coincide in the world frame in the initial configuration.
    """

    body_a: int
    body_b: int
    anchor_a: np.ndarray
    anchor_b: np.ndarray
    axis_a: np.ndarray
    axis_b: np.ndarray

    def __post_init__(self) -> None:
        self.anchor_a = _vec(self.anchor_a, 3)
        self.anchor_b = _vec(self.anchor_b, 3)
        self.axis_a = _vec(self.axis_a, 3)
        self.axis_b = _vec(self.axis_b, 3)
        if np.linalg.norm(self.axis_a) == 0.0 or np.linalg.norm(self.axis_b) == 0.0:
            raise ValueError("hinge axes must be non-zero")


@dataclass
class FixedJoint:
    """Weld: the anchors coincide and the relative orientation stays at its initial value."""

    body_a: int
    body_b: int
    anchor_a: np.ndarray
    anchor_b: np.ndarray

    def __post_init__(self) -> None:
        self.anchor_a = _vec(self.anchor_a, 3)
        self.anchor_b = _vec(self.anchor_b, 3)


@dataclass
class PrismaticJoint:
    """Slider: the relative orientation is fixed and body b slides along ``axis_a``.

    ``axis_a`` is the slide axis in the frame of body a (the world frame for ``WORLD``); the
    anchors mark the points whose displacement must stay along it, so initially the world
    displacement between them must be parallel to the axis.
    """

    body_a: int
    body_b: int
    anchor_a: np.ndarray
    anchor_b: np.ndarray
    axis_a: np.ndarray

    def __post_init__(self) -> None:
        self.anchor_a = _vec(self.anchor_a, 3)
        self.anchor_b = _vec(self.anchor_b, 3)
        self.axis_a = _vec(self.axis_a, 3)
        if np.linalg.norm(self.axis_a) == 0.0:
            raise ValueError("prismatic axis must be non-zero")


@dataclass
class SphereCollider:
    """Sphere centred at the body's centre of mass (attached to a body entity).

    ``restitution`` is the Newton restitution coefficient of the material; a contact uses the
    smaller of the two coefficients.
    """

    radius: float
    restitution: float = 0.0

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")


@dataclass
class PlaneCollider:
    """Static half-space ``normal . x >= offset`` in the world frame (the world body)."""

    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    offset: float = 0.0
    restitution: float = 0.0

    def __post_init__(self) -> None:
        normal = _vec(self.normal, 3)
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            raise ValueError("plane normal must be non-zero")
        self.normal = normal / norm
        self.offset = float(self.offset)
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")


@dataclass
class Gravity:
    acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))

    def __post_init__(self) -> None:
        self.acceleration = _vec(self.acceleration, 3)


@dataclass
class SimulationConfig:
    dt: float = 1e-3
    theta: float = 0.5
    # Velocity-level position stabilization: g_dot = -(stabilization / dt) * g
    stabilization: float = 0.2
    # Normal contacts (see docs/theory.md, "Contacts")
    max_contacts: int | None = None  # contact buffer capacity; None: 8 per collider
    contact_iterations: int = 50  # block-iteration cap per step
    contact_tolerance: float = 1e-9  # stop when the impulse changes by less than this
    contact_omega: float = 1.0  # relaxation of the projected Jacobi update
    contact_stabilization: float = 0.0  # penetration recovery gain, like ``stabilization``
    restitution_threshold: float = 1e-2  # approach speed below which e is treated as 0

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if not 0.0 < self.theta <= 1.0:
            raise ValueError("theta must be in (0, 1]")
        if self.max_contacts is not None and self.max_contacts < 1:
            raise ValueError("max_contacts must be positive")
        if self.contact_iterations < 1:
            raise ValueError("contact_iterations must be positive")
        if not 0.0 < self.contact_omega <= 1.0:
            raise ValueError("contact_omega must be in (0, 1]")
