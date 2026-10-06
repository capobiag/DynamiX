"""ECS layer: scene description only (topology, configuration)."""

from dynamix.ecs.components import BallJoint, Body, Gravity, SimulationConfig
from dynamix.ecs.scene import WORLD, Scene

__all__ = ["WORLD", "BallJoint", "Body", "Gravity", "Scene", "SimulationConfig"]
