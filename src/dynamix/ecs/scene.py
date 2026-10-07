"""Scene description backed by an esper world."""

from __future__ import annotations

import itertools

import esper

from dynamix.ecs.components import BallJoint, Body, Gravity, HingeJoint, SimulationConfig

# Sentinel entity id for the fixed world frame in joints.
WORLD = -1

_counter = itertools.count()


class Scene:
    """Thin wrapper owning a private esper world (esper 3 uses named global worlds)."""

    def __init__(self, config: SimulationConfig | None = None, gravity: Gravity | None = None):
        self.name = f"dynamix-scene-{next(_counter)}"
        esper.switch_world(self.name)
        self.config = config or SimulationConfig()
        self.gravity = gravity or Gravity()
        self.config_entity = esper.create_entity(self.config, self.gravity)

    def _activate(self) -> None:
        esper.switch_world(self.name)

    def add_body(self, body: Body) -> int:
        self._activate()
        return esper.create_entity(body)

    def add_ball_joint(self, joint: BallJoint) -> int:
        return self._add_joint(joint)

    def add_hinge_joint(self, joint: HingeJoint) -> int:
        return self._add_joint(joint)

    def _add_joint(self, joint: BallJoint | HingeJoint) -> int:
        self._activate()
        for ent in (joint.body_a, joint.body_b):
            if ent != WORLD and not esper.has_component(ent, Body):
                raise ValueError(f"entity {ent} is not a body")
        if joint.body_a == joint.body_b:
            raise ValueError("a joint must connect two distinct bodies")
        if joint.body_b == WORLD:
            raise ValueError("body_b must be a body; use body_a for the world")
        return esper.create_entity(joint)

    def bodies(self) -> list[tuple[int, Body]]:
        self._activate()
        return sorted(esper.get_component(Body), key=lambda item: item[0])

    def joints(self) -> list[tuple[int, BallJoint | HingeJoint]]:
        """All joints in creation order, regardless of kind."""
        self._activate()
        found = esper.get_component(BallJoint) + esper.get_component(HingeJoint)
        return sorted(found, key=lambda item: item[0])

    def close(self) -> None:
        esper.delete_world(self.name)
