"""JAX backend vs. the NumPy reference backend."""

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import dynamix.jax_backend as jb  # noqa: E402
from dynamix.core import compile_scene  # noqa: E402
from dynamix.ecs import WORLD, BallJoint, Body, HingeJoint, Scene  # noqa: E402
from dynamix.jax_backend import quaternion as jquat  # noqa: E402
from dynamix.jax_backend.banded import (  # noqa: E402
    block_band_to_dense,
    solve_spd_block_banded,
)
from dynamix.numpy_backend import Engine as NumpyEngine  # noqa: E402
from dynamix.numpy_backend import quaternion as nquat  # noqa: E402
from dynamix.scenes import build_chain  # noqa: E402


def buffers(n):
    return compile_scene(build_chain(n))


def random_spd_band(m, nj, seed=0):
    """Random SPD matrix with block half-bandwidth m, returned in band storage and dense."""
    rng = np.random.default_rng(seed)
    dense = np.zeros((3 * nj, 3 * nj))
    for i in range(nj):
        for j in range(max(0, i - m), i + 1):
            blk = rng.normal(size=(3, 3)) * 0.3
            dense[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = blk
    dense = dense + dense.T
    dense += np.eye(3 * nj) * (np.abs(dense).sum(axis=1).max() + 1.0)
    band = np.zeros((m + 1, nj, 3, 3))
    for d in range(m + 1):
        for j in range(nj - d):
            band[d, j] = dense[3 * (j + d) : 3 * (j + d) + 3, 3 * j : 3 * j + 3]
    return band, dense


@pytest.mark.parametrize("n", [1, 3, 10, 11, 40])
def test_chain_matches_numpy(n):
    np_engine = NumpyEngine(b := buffers(n))
    jax_engine = jb.Engine(buffers(n))
    np_engine.run(400)
    jax_engine.run(400)
    np.testing.assert_allclose(np.asarray(jax_engine.state.q), b.q, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jax_engine.state.u), b.u, atol=1e-8)
    assert jax_engine.time == pytest.approx(400 * b.dt)


def test_step_and_run_agree():
    a, b = jb.Engine(buffers(5)), jb.Engine(buffers(5))
    for _ in range(50):
        a.step()
    b.run(50)
    np.testing.assert_allclose(np.asarray(a.state.q), np.asarray(b.state.q), atol=1e-12)


def test_rollout_frames():
    engine = jb.Engine(buffers(3))
    frames = engine.rollout(100, stride=10)
    assert frames.q.shape == (10, 3, 7)
    reference = jb.Engine(buffers(3))
    reference.run(100)
    np.testing.assert_allclose(np.asarray(frames.q[-1]), np.asarray(reference.state.q), atol=1e-12)


def test_branched_unordered_topology_matches_numpy():
    base = buffers(4)
    # Bodies 2 and 3 both hang from body 1 (a tree), and joints are numbered out of order.
    joint_a = base.joint_a.copy()
    joint_a[3] = 1
    perm = np.array([2, 0, 3, 1])
    branched = dataclasses.replace(
        base,
        joint_a=joint_a[perm],
        joint_b=base.joint_b[perm],
        joint_anchor_a=base.joint_anchor_a[perm],
        joint_anchor_b=base.joint_anchor_b[perm],
    )
    copy = dataclasses.replace(branched, q=branched.q.copy(), u=branched.u.copy())
    NumpyEngine(copy).run(100)
    jax_engine = jb.Engine(branched)
    jax_engine.run(100)
    np.testing.assert_allclose(np.asarray(jax_engine.state.q), copy.q, atol=1e-9)


@pytest.mark.parametrize("m", [0, 1, 2])
def test_block_banded_solve_matches_dense(m):
    band, dense = random_spd_band(m, 9)
    rhs = np.random.default_rng(1).normal(size=27)
    np.testing.assert_allclose(np.asarray(block_band_to_dense(jnp.asarray(band))), dense)
    x = solve_spd_block_banded(jnp.asarray(band), jnp.asarray(rhs))
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(dense, rhs), atol=1e-10)


def test_quaternion_matches_numpy():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(6, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w = rng.normal(size=(6, 3))
    np.testing.assert_allclose(np.asarray(jquat.to_matrix(q)), nquat.to_matrix(q), atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(jquat.integrate(q, w, 0.01)), nquat.integrate(q, w, 0.01), atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(jquat.multiply(q, q[::-1])), nquat.multiply(q, q[::-1]), atol=1e-14
    )


def test_vmap_over_initial_states():
    engine = jb.Engine(buffers(4))
    kw = engine._kw
    batch = jax.tree_util.tree_map(lambda x: jnp.stack((x, x, x)), engine.state)
    batch = batch._replace(u=batch.u.at[1, :, 3].add(0.5))
    out = jax.vmap(lambda s: jb.run(s, engine.params, engine.system, steps=20, **kw))(batch)
    single = jb.run(engine.state, engine.params, engine.system, steps=20, **kw)
    np.testing.assert_allclose(np.asarray(out.q[0]), np.asarray(single.q), atol=1e-12)
    assert not np.allclose(np.asarray(out.q[1]), np.asarray(single.q))


def test_grad_matches_finite_differences():
    engine = jb.Engine(buffers(3))
    kw = engine._kw

    def loss(angular):
        s = engine.state._replace(u=engine.state.u.at[0, 3].set(angular))
        return jb.run(s, engine.params, engine.system, steps=30, **kw).q[-1, 2]

    x0, eps = 0.3, 1e-6
    fd = (loss(x0 + eps) - loss(x0 - eps)) / (2 * eps)
    assert float(jax.grad(loss)(x0)) == pytest.approx(float(fd), rel=1e-5, abs=1e-8)


def test_step_does_not_retrace():
    engine = jb.Engine(buffers(3))
    engine.step()
    before = jb.step._cache_size()
    for _ in range(5):
        engine.step()
    assert jb.step._cache_size() == before


def mixed_buffers():
    scene = Scene()
    a = scene.add_body(Body(mass=1.0, inertia=[0.1, 0.2, 0.3], position=[0.5, 0, 0]))
    b = scene.add_body(Body(mass=2.0, inertia=[0.1, 0.1, 0.1], position=[1.5, 0, 0]))
    c = scene.add_body(Body(mass=1.0, inertia=[0.2, 0.1, 0.1], position=[2.5, 0, 0]))
    scene.add_ball_joint(BallJoint(WORLD, a, [0, 0, 0], [-0.5, 0, 0]))
    scene.add_hinge_joint(HingeJoint(a, b, [0.5, 0, 0], [-0.5, 0, 0], [0, 1, 0], [0, 1, 0]))
    scene.add_hinge_joint(HingeJoint(b, c, [0.5, 0, 0], [-0.5, 0, 0], [0, 1, 0], [0, 1, 0]))
    buffers = compile_scene(scene)
    buffers.u[:, 3:] = [0.3, 0.2, -0.4]
    return buffers


@pytest.mark.parametrize("n", [3, 15])
def test_hinge_chain_matches_numpy(n):
    np_engine = NumpyEngine(b := compile_scene(build_chain(n, joint="hinge")))
    jax_engine = jb.Engine(compile_scene(build_chain(n, joint="hinge")))
    np_engine.run(300)
    jax_engine.run(300)
    np.testing.assert_allclose(np.asarray(jax_engine.state.q), b.q, atol=1e-9)


def test_mixed_joints_match_numpy():
    b = mixed_buffers()
    jax_engine = jb.Engine(mixed_buffers())
    NumpyEngine(b).run(300)
    jax_engine.run(300)
    np.testing.assert_allclose(np.asarray(jax_engine.state.q), b.q, atol=1e-9)
    np.testing.assert_allclose(np.asarray(jax_engine.state.u), b.u, atol=1e-8)
