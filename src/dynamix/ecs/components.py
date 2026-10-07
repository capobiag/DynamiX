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

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if not 0.0 < self.theta <= 1.0:
            raise ValueError("theta must be in (0, 1]")
