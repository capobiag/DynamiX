"""JAX contacts: detection and dynamics must agree with the NumPy backend."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import dynamix.jax_backend as jb  # noqa: E402
from dynamix.core import compile_scene  # noqa: E402
from dynamix.ecs import (  # noqa: E402
    WORLD,
    BallJoint,
    Body,
    Gravity,
    PlaneCollider,
    Scene,
    SimulationConfig,
    SphereCollider,
)
from dynamix.jax_backend import contacts as bc  # noqa: E402
from dynamix.numpy_backend import Engine  # noqa: E402
from dynamix.numpy_backend.contacts import ContactDetector  # noqa: E402
from dynamix.scenes import build_ball_pile, build_bouncing_ball  # noqa: E402


def contact_set(c, decimals=10):
    """Order-independent description of the valid contacts of a buffer."""
    k = int(c.count)
    a, b = np.asarray(c.body_a)[:k], np.asarray(c.body_b)[:k]
    key = np.lexsort((np.round(np.asarray(c.gap)[:k], decimals), b, a))
    arrays = (a, b, np.asarray(c.gap)[:k], np.asarray(c.normal)[:k], np.asarray(c.point)[:k])
    arrays += (np.asarray(c.restitution)[:k],)
    return [x[key] for x in arrays]


def mixed_scene(n=40, capacity=None, seed=0):
    rng = np.random.default_rng(seed)
    scene = Scene(SimulationConfig(max_contacts=capacity), Gravity(np.array([0.0, 0.0, -9.81])))
    for _ in range(n):
        radius = rng.uniform(0.08, 0.15)
        body = scene.add_body(
            Body(
                mass=rng.uniform(0.5, 2.0),
                inertia=np.full(3, 0.01),
                position=rng.uniform(0.0, 0.8, 3),
                linear_velocity=rng.normal(size=3),
            )
        )
        scene.add_sphere_collider(body, SphereCollider(radius, rng.uniform(0.0, 1.0)))
    scene.add_plane(PlaneCollider(normal=np.array([0.0, 0.3, 1.0]), offset=0.1, restitution=0.6))
    scene.add_plane(PlaneCollider(normal=np.array([1.0, 0.0, 0.0]), offset=0.05, restitution=0.2))
    return scene


def jax_detect(buf, slots=4):
    params = jb.make_params(buf)[0].contacts
    cfg = jb.make_contact_settings(buf, slots)
    return bc.detect(params, cfg, jax.numpy.asarray(buf.q))


def test_detection_matches_numpy_with_mixed_radii_and_several_planes():
    buf = compile_scene(mixed_scene())
    expected = ContactDetector(buf).detect(buf.q)
    got = jax_detect(buf, slots=8)
    assert int(got.count) == expected.count > 5 and not bool(got.overflow)
    for e, g in zip(contact_set(expected), contact_set(got), strict=True):
        np.testing.assert_allclose(g, e, atol=1e-12)


def test_capacity_overflow_keeps_the_deepest_contacts_and_flags():
    full = compile_scene(mixed_scene())
    n_all = ContactDetector(full).detect(full.q).count
    capacity = n_all - 4
    buf = compile_scene(mixed_scene(capacity=capacity))
    with pytest.warns(UserWarning, match="overflow"):
        expected = ContactDetector(buf).detect(buf.q)
    got = jax_detect(buf, slots=8)
    assert bool(got.overflow) and int(got.count) == capacity == expected.count
    np.testing.assert_allclose(
        np.sort(np.asarray(got.gap)), np.sort(expected.gap[:capacity]), atol=1e-12
    )


def test_grid_cell_overflow_is_flagged():
    scene = Scene(SimulationConfig(), Gravity(np.zeros(3)))
    for k in range(6):  # six spheres in one cell, more than 2 slots can see
        body = scene.add_body(Body(1.0, np.full(3, 0.01), np.array([0.01 * k, 0.0, 0.0])))
        scene.add_sphere_collider(body, SphereCollider(0.5))
    buf = compile_scene(scene)
    assert bool(jax_detect(buf, slots=2).overflow)
    assert not bool(jax_detect(buf, slots=6).overflow)


def run_both(scene, steps):
    np_buf, jx_buf = compile_scene(scene), compile_scene(scene)
    numpy_engine, jax_engine = Engine(np_buf), jb.Engine(jx_buf)
    numpy_engine.run(steps)
    jax_engine.run(steps)
    return np_buf, jax_engine.state


def test_bouncing_ball_matches_numpy():
    np_buf, state = run_both(build_bouncing_ball(1.0, 0.1, 1.0, 0.8), 1500)
    np.testing.assert_allclose(np.asarray(state.q), np_buf.q, atol=1e-10)
    np.testing.assert_allclose(np.asarray(state.u), np_buf.u, atol=1e-10)


def test_ball_pile_matches_numpy():
    config = SimulationConfig(max_contacts=200)
    np_buf, state = run_both(build_ball_pile(64, config=config), 400)
    np.testing.assert_allclose(np.asarray(state.q), np_buf.q, atol=1e-8)
    np.testing.assert_allclose(np.asarray(state.u), np_buf.u, atol=1e-8)


def test_joint_and_contact_match_numpy():
    scene = Scene(SimulationConfig(), Gravity(np.array([0.0, 0.0, -9.81])))
    length, radius = 1.0, 0.1
    bob = scene.add_body(Body(1.0, np.full(3, 0.004), np.array([length, 0.0, 0.0])))
    scene.add_sphere_collider(bob, SphereCollider(radius, 0.5))
    scene.add_ball_joint(BallJoint(WORLD, bob, np.zeros(3), np.array([-length, 0.0, 0.0])))
    scene.add_plane(PlaneCollider(offset=-length + 0.1, restitution=0.5))
    np_buf, state = run_both(scene, 2000)
    np.testing.assert_allclose(np.asarray(state.q), np_buf.q, atol=1e-8)
    np.testing.assert_allclose(np.asarray(state.u), np_buf.u, atol=1e-8)
    assert abs(np.linalg.norm(np.asarray(state.q)[0, :3]) - length) < 1e-3


def test_scan_matches_single_steps_and_vmap_works():
    buf = compile_scene(build_ball_pile(27, config=SimulationConfig(max_contacts=100)))
    engine = jb.Engine(buf)
    kw = engine._kw
    state = engine.state
    stepped = state
    for _ in range(20):
        stepped = jb.step(stepped, engine.params, engine.system, **kw)
    scanned = jb.run(state, engine.params, engine.system, steps=20, **kw)
    np.testing.assert_allclose(np.asarray(scanned.q), np.asarray(stepped.q), atol=1e-12)

    batch = jax.tree_util.tree_map(lambda x: jax.numpy.stack([x, x]), state)
    batched = jax.vmap(lambda s: jb.step(s, engine.params, engine.system, **kw))(batch)
    single = jb.step(state, engine.params, engine.system, **kw)
    np.testing.assert_allclose(np.asarray(batched.q[1]), np.asarray(single.q), atol=1e-12)


def test_scenes_without_colliders_have_no_contact_stage():
    from dynamix.scenes import build_pendulum

    engine = jb.Engine(compile_scene(build_pendulum()))
    assert engine.contact_cfg is None and engine.params.contacts is None
    assert engine.contacts is None


def test_long_chain_with_contacts_uses_banded_solve_and_matches_numpy():
    from dynamix.scenes import build_chain

    scene = build_chain(14, angles=np.pi / 2)  # more joints than the dense-solve threshold
    for body, _ in scene.bodies():
        scene.add_sphere_collider(body, SphereCollider(0.1, 0.3))
    scene.add_plane(PlaneCollider(offset=-4.0, restitution=0.3))
    np_buf, state = run_both(scene, 1500)
    np.testing.assert_allclose(np.asarray(state.q), np_buf.q, atol=1e-6)
    np.testing.assert_allclose(np.asarray(state.u), np_buf.u, atol=1e-6)
    assert np.asarray(state.q)[:, 2].min() > -4.0 - 0.05
