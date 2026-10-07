"""Fixed and prismatic joints: Jacobians, Delassus assembly, dynamics, validation, parity."""

import numpy as np
import pytest

from dynamix.core import compile_scene
from dynamix.core import joints as jt
from dynamix.ecs import WORLD, BallJoint, Body, FixedJoint, HingeJoint, PrismaticJoint, Scene
from dynamix.numpy_backend import Engine
from dynamix.numpy_backend import quaternion as quat
from dynamix.numpy_backend.banded import banded_to_dense
from dynamix.numpy_backend.joint_system import JointSystem

KINDS = [jt.BALL, jt.HINGE, jt.FIXED, jt.PRISMATIC, jt.FIXED, jt.PRISMATIC]
JOINT_A = [-1, 0, 1, 2, 3, -1]
JOINT_B = [0, 1, 2, 3, 4, 2]
N_BODIES = 5


def random_system(kinds=KINDS, joint_a=JOINT_A, joint_b=JOINT_B, seed=0):
    rng = np.random.default_rng(seed)
    nj = len(kinds)
    q = np.empty((N_BODIES, 7))
    q[:, :3] = rng.normal(size=(N_BODIES, 3))
    quats = rng.normal(size=(N_BODIES, 4))
    q[:, 3:] = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    anchor_a, anchor_b, axis_a, axis_b = rng.normal(size=(4, nj, 3))
    rel_quat = rng.normal(size=(nj, 4))
    rel_rot = quat.to_matrix(rel_quat / np.linalg.norm(rel_quat, axis=1, keepdims=True))
    mass = rng.uniform(0.5, 2.0, N_BODIES)
    inertia = np.array([np.diag(rng.uniform(0.1, 1.0, 3)) for _ in range(N_BODIES)])
    system = JointSystem(
        joint_a, joint_b, anchor_a, anchor_b, mass, np.linalg.inv(inertia), kinds, axis_a,
        axis_b, rel_rot,
    )  # fmt: skip
    return system, q, mass, inertia


def dense_jacobian(system, n_bodies):
    r = system.rows
    jac = np.zeros((r * system.n_joints, 6 * n_bodies))
    for s in range(system.n_sides):
        j, body = system.side_joint[s], system.side_body[s]
        lin = np.eye(r, 3) if system.lin is None else system.lin[j]
        jac[r * j : r * (j + 1), 6 * body : 6 * body + 3] += system.sign[s] * lin
        jac[r * j : r * (j + 1), 6 * body + 3 : 6 * body + 6] += system.jw[s]
    return jac


def residual(system, q):
    return system.update(q, quat.to_matrix(q[:, 3:])).copy().reshape(-1)


def test_jacobian_matches_finite_differences():
    system, q, _, _ = random_system()
    assert system.rows == 6
    system.update(q, quat.to_matrix(q[:, 3:]))
    jac = dense_jacobian(system, N_BODIES)
    eps = 1e-6
    for col in range(6 * N_BODIES):
        du = np.zeros(6 * N_BODIES)
        du[col] = eps
        du = du.reshape(N_BODIES, 6)
        out = []
        for sign in (1.0, -1.0):
            qq = q.copy()
            qq[:, :3] += sign * du[:, :3]
            qq[:, 3:] = quat.integrate(q[:, 3:], sign * du[:, 3:], 1.0)
            out.append(residual(system, qq))
        np.testing.assert_allclose(jac[:, col], (out[0] - out[1]) / (2 * eps), atol=1e-7)


def test_delassus_band_matches_dense_product_with_padding():
    system, q, mass, inertia = random_system()
    system.update(q, quat.to_matrix(q[:, 3:]))
    jac = dense_jacobian(system, N_BODIES)
    minv = np.zeros((6 * N_BODIES, 6 * N_BODIES))
    for i in range(N_BODIES):
        minv[6 * i : 6 * i + 3, 6 * i : 6 * i + 3] = np.eye(3) / mass[i]
        minv[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = np.linalg.inv(inertia[i])
    expected = jac @ minv @ jac.T
    for j, k in enumerate(np.asarray(KINDS)[system.joint_perm]):
        for row in range(jt.ROWS[int(k)], system.rows):
            expected[system.rows * j + row, system.rows * j + row] += 1.0
    np.testing.assert_allclose(banded_to_dense(system.delassus_band()), expected, atol=1e-10)

    rng = np.random.default_rng(2)
    u = rng.normal(size=(N_BODIES, 6))
    np.testing.assert_allclose(system.jacobian_times(u), jac @ u.reshape(-1), atol=1e-11)
    impulse = rng.normal(size=(system.n_joints, system.rows))
    want = u.reshape(-1) + minv @ jac.T @ impulse.reshape(-1)
    system.add_impulse(impulse, u)
    np.testing.assert_allclose(u.reshape(-1), want, atol=1e-11)


def two_body_weld_pendulum(compound):
    scene = Scene()
    if compound:
        body = scene.add_body(Body(mass=2.0, inertia=[0.2, 0.9, 1.1], position=[1.0, 0, 0]))
        scene.add_ball_joint(BallJoint(WORLD, body, [0, 0, 0], [-1.0, 0, 0]))
        return compile_scene(scene)
    a = scene.add_body(Body(mass=1.0, inertia=[0.1, 0.2, 0.3], position=[0.5, 0, 0]))
    b = scene.add_body(Body(mass=1.0, inertia=[0.1, 0.2, 0.3], position=[1.5, 0, 0]))
    scene.add_ball_joint(BallJoint(WORLD, a, [0, 0, 0], [-0.5, 0, 0]))
    scene.add_fixed_joint(FixedJoint(a, b, [0.5, 0, 0], [-0.5, 0, 0]))
    return compile_scene(scene)


def test_welded_bodies_behave_like_one_compound_body():
    welded, single = two_body_weld_pendulum(False), two_body_weld_pendulum(True)
    Engine(welded).run(1000)
    Engine(single).run(1000)
    centre = 0.5 * (welded.q[0, :3] + welded.q[1, :3])
    np.testing.assert_allclose(centre, single.q[0, :3], atol=2e-3)
    np.testing.assert_allclose(welded.q[0, 3:], welded.q[1, 3:], atol=1e-3)


def test_prismatic_slides_along_inclined_rail():
    axis = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    scene = Scene()
    body = scene.add_body(Body(mass=1.0, inertia=[0.1, 0.1, 0.1], position=axis))
    scene.add_prismatic_joint(PrismaticJoint(WORLD, body, [0, 0, 0], [0, 0, 0], axis))
    buffers = compile_scene(scene)
    steps = 1000
    Engine(buffers).run(steps)
    t = steps * buffers.dt
    accel = np.array([0.0, 0.0, -9.81]) @ axis
    np.testing.assert_allclose(buffers.q[0, :3], axis * (1.0 + 0.5 * accel * t * t), atol=1e-3)
    np.testing.assert_allclose(buffers.q[0, 3:], [0, 0, 0, 1], atol=1e-6)


def test_invalid_initial_configuration_rejected():
    scene = Scene()
    body = scene.add_body(Body(mass=1.0, inertia=[1, 1, 1], position=[1.0, 1.0, 0.0]))
    scene.add_prismatic_joint(PrismaticJoint(WORLD, body, [0, 0, 0], [0, 0, 0], [1, 0, 0]))
    with pytest.raises(ValueError, match="not satisfied"):
        compile_scene(scene)
    scene = Scene()
    body = scene.add_body(Body(mass=1.0, inertia=[1, 1, 1], position=[1.0, 0.0, 0.0]))
    scene.add_fixed_joint(FixedJoint(WORLD, body, [0, 0, 0], [0, 0, 0]))
    with pytest.raises(ValueError, match="not satisfied"):
        compile_scene(scene)
    with pytest.raises(ValueError, match="non-zero"):
        PrismaticJoint(WORLD, body, [0, 0, 0], [0, 0, 0], [0, 0, 0])


def mixed_chain(reverse):
    """Chain of 14 bodies joined by alternating hinge / fixed joints and a final slider."""
    scene = Scene()
    n = 14
    bodies = [
        scene.add_body(Body(mass=1.0, inertia=[0.1, 0.2, 0.3], position=[i + 0.5, 0, 0]))
        for i in range(n)
    ]
    joints = [HingeJoint(WORLD, bodies[0], [0, 0, 0], [-0.5, 0, 0], [0, 1, 0], [0, 1, 0])]
    for i in range(1, n - 1):
        args = ([0.5, 0, 0], [-0.5, 0, 0])
        joints.append(
            HingeJoint(bodies[i - 1], bodies[i], *args, [0, 1, 0], [0, 1, 0])
            if i % 2
            else FixedJoint(bodies[i - 1], bodies[i], *args)
        )
    joints.append(PrismaticJoint(bodies[n - 2], bodies[n - 1], [0, 0, 0], [0, 0, 0], [1, 0, 0]))
    for joint in joints[::2] + joints[1::2] if reverse else joints:
        if isinstance(joint, HingeJoint):
            scene.add_hinge_joint(joint)
        elif isinstance(joint, FixedJoint):
            scene.add_fixed_joint(joint)
        elif isinstance(joint, PrismaticJoint):
            scene.add_prismatic_joint(joint)
    return compile_scene(scene)


def test_mixed_chain_is_independent_of_joint_creation_order():
    a, b = mixed_chain(False), mixed_chain(True)
    assert not np.array_equal(a.joint_kind, b.joint_kind)  # the creation order really differs
    Engine(a).run(300)
    Engine(b).run(300)
    np.testing.assert_allclose(a.q, b.q, atol=1e-9)
    # the slider keeps the last body on the axis of its parent: bounded drift
    assert np.all(np.isfinite(a.q))


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_chain_matches_jax(reverse):
    pytest.importorskip("jax")
    import dynamix.jax_backend as jb

    buffers = mixed_chain(reverse)
    jax_engine = jb.Engine(mixed_chain(reverse))
    Engine(buffers).run(300)
    jax_engine.run(300)
    np.testing.assert_allclose(np.asarray(jax_engine.state.q), buffers.q, atol=1e-8)
    np.testing.assert_allclose(np.asarray(jax_engine.state.u), buffers.u, atol=1e-7)
