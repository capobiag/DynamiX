"""ECS layer: scene description only (topology, configuration)."""

from dynamix.ecs.components import (
    BallJoint,
    Body,
    FixedJoint,
    Gravity,
    HingeJoint,
    PrismaticJoint,
    SimulationConfig,
)
from dynamix.ecs.scene import WORLD, Scene

__all__ = [
    "WORLD",
    "BallJoint",
    "Body",
    "FixedJoint",
    "Gravity",
    "HingeJoint",
    "PrismaticJoint",
    "Scene",
    "SimulationConfig",
]
