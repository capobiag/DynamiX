"""Scene description backed by an esper world."""

from __future__ import annotations

import itertools

import esper

from dynamix.ecs.components import (
    BallJoint,
    Body,
    FixedJoint,
    Gravity,
    HingeJoint,
    PlaneCollider,
    PrismaticJoint,
    SimulationConfig,
    SphereCollider,
)

Joint = BallJoint | HingeJoint | FixedJoint | PrismaticJoint
JOINT_TYPES = (BallJoint, HingeJoint, FixedJoint, PrismaticJoint)

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

    def add_fixed_joint(self, joint: FixedJoint) -> int:
        return self._add_joint(joint)

    def add_prismatic_joint(self, joint: PrismaticJoint) -> int:
        return self._add_joint(joint)

    def _add_joint(self, joint: Joint) -> int:
        self._activate()
        for ent in (joint.body_a, joint.body_b):
            if ent != WORLD and not esper.has_component(ent, Body):
                raise ValueError(f"entity {ent} is not a body")
        if joint.body_a == joint.body_b:
            raise ValueError("a joint must connect two distinct bodies")
        if joint.body_b == WORLD:
            raise ValueError("body_b must be a body; use body_a for the world")
        return esper.create_entity(joint)

    def add_sphere_collider(self, body: int, collider: SphereCollider) -> None:
        self._activate()
        if not esper.has_component(body, Body):
            raise ValueError(f"entity {body} is not a body")
        if esper.has_component(body, SphereCollider):
            raise ValueError(f"body {body} already has a collider")
        esper.add_component(body, collider)

    def add_plane(self, plane: PlaneCollider) -> int:
        self._activate()
        return esper.create_entity(plane)

    def sphere_colliders(self) -> dict[int, SphereCollider]:
        self._activate()
        return dict(esper.get_component(SphereCollider))

    def planes(self) -> list[PlaneCollider]:
        self._activate()
        return [p for _, p in sorted(esper.get_component(PlaneCollider), key=lambda x: x[0])]

    def bodies(self) -> list[tuple[int, Body]]:
        self._activate()
        return sorted(esper.get_component(Body), key=lambda item: item[0])

    def joints(self) -> list[tuple[int, Joint]]:
        """All joints in creation order, regardless of kind."""
        self._activate()
        found = [x for kind in JOINT_TYPES for x in esper.get_component(kind)]
        return sorted(found, key=lambda item: item[0])

    def close(self) -> None:
        esper.delete_world(self.name)
