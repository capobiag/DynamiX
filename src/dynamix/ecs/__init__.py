"""ECS layer: scene description only (topology, configuration)."""

from dynamix.ecs.components import BallJoint, Body, Gravity, HingeJoint, SimulationConfig
from dynamix.ecs.scene import WORLD, Scene

__all__ = ["WORLD", "BallJoint", "Body", "Gravity", "HingeJoint", "Scene", "SimulationConfig"]
