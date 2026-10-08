"""Hinge joint: Jacobian vs. finite differences, mixed joint systems, planar-chain reference."""

import numpy as np
import pytest

from dynamix.core import compile_scene
from dynamix.core import joints as jt
from dynamix.ecs import WORLD, BallJoint, Body, HingeJoint, Scene
from dynamix.numpy_backend import Engine
from dynamix.numpy_backend import quaternion as quat
from dynamix.numpy_backend.joint_system import JointSystem
from dynamix.scenes import build_chain
from reference import simulate_chain

# joint 0: hinge world-body0, joint 1: ball body0-body1, joint 2: hinge body1-body2
JOINTS = {
    "joint_a": [-1, 0, 1],
    "joint_b": [0, 1, 2],
    "kind": [jt.HINGE, jt.BALL, jt.HINGE],
}


def random_setup(seed=0):
    rng = np.random.default_rng(seed)
    n = 3
    q = np.empty((n, 7))
    q[:, :3] = rng.normal(size=(n, 3))
    quats = rng.normal(size=(n, 4))
    q[:, 3:] = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    anchor_a, anchor_b = rng.normal(size=(2, 3, 3))
    axis_a, axis_b = rng.normal(size=(2, 3, 3))
    system = JointSystem(
        JOINTS["joint_a"], JOINTS["joint_b"], anchor_a, anchor_b, np.ones(n),
        np.tile(np.eye(3), (n, 1, 1)), JOINTS["kind"], axis_a, axis_b,
    )  # fmt: skip
    return system, q, (anchor_a, anchor_b, axis_a, axis_b)


def hinge_residual_reference(q, joint, anchor_a, anchor_b, axis_a, axis_b):
    """Plain per-joint formula (independent of ``dynamix.core.joints``)."""
    a, b = joint
    rot = quat.to_matrix(q[:, 3:])
    ra, xa = (np.eye(3), np.zeros(3)) if a < 0 else (rot[a], q[a, :3])
    point = (q[b, :3] + rot[b] @ anchor_b) - (xa + ra @ anchor_a)
    axis_a, axis_b = axis_a / np.linalg.norm(axis_a), axis_b / np.linalg.norm(axis_b)
    helper = np.eye(3)[np.argmin(np.abs(axis_a))]
    p1 = np.cross(axis_a, helper)
    p1 /= np.linalg.norm(p1)
    p2 = np.cross(axis_a, p1)
    c = rot[b] @ axis_b
    return np.concatenate((point, [(ra @ p1) @ c, (ra @ p2) @ c]))


def dense_jacobian(system, n_bodies):
    jac = np.zeros((system.rows * system.n_joints, 6 * n_bodies))
    for s in range(system.n_sides):
        rows = slice(system.rows * system.side_joint[s], system.rows * (system.side_joint[s] + 1))
        body = system.side_body[s]
        jac[rows, 6 * body : 6 * body + 3] = system.sign[s] * np.eye(system.rows, 3)
        jac[rows, 6 * body + 3 : 6 * body + 6] = system.jw[s]
    return jac


def test_hinge_jacobian_matches_finite_differences():
    system, q, (anchor_a, anchor_b, axis_a, axis_b) = random_setup()
    rot = quat.to_matrix(q[:, 3:])
    system.update(q, rot)
    jac = dense_jacobian(system, 3)

    def residual(qq):
        out = np.zeros((3, 5))
        for j, (a, b) in enumerate(zip(JOINTS["joint_a"], JOINTS["joint_b"], strict=True)):
            if JOINTS["kind"][j] == jt.HINGE:
                out[j] = hinge_residual_reference(
                    qq, (a, b), anchor_a[j], anchor_b[j], axis_a[j], axis_b[j]
                )
            else:
                rr = quat.to_matrix(qq[:, 3:])
                xa, ra = (np.zeros(3), np.eye(3)) if a < 0 else (qq[a, :3], rr[a])
                out[j, :3] = (qq[b, :3] + rr[b] @ anchor_b[j]) - (xa + ra @ anchor_a[j])
        return out

    # the system residual agrees with the plain formulas (ball rows are zero-padded)
    np.testing.assert_allclose(system.g, residual(q), atol=1e-13)

    eps = 1e-6
    for col in range(18):
        du = np.zeros(18)
        du[col] = 1.0
        plus, minus = q.copy(), q.copy()
        for sign, qq in ((1, plus), (-1, minus)):
            d = sign * eps * du.reshape(3, 6)
            qq[:, :3] += d[:, :3]
            qq[:, 3:] = quat.integrate(q[:, 3:], d[:, 3:], 1.0)
        fd = (residual(plus) - residual(minus)).reshape(-1) / (2 * eps)
        np.testing.assert_allclose(jac[:, col], fd, atol=1e-8)


def test_mixed_system_band_matches_dense_delassus():
    system, q, _ = random_setup(1)
    system.update(q, quat.to_matrix(q[:, 3:]))
    band = system.delassus_band()
    jac = dense_jacobian(system, 3)
    expected = jac @ jac.T  # unit mass and inertia
    # the ball joint (joint 1) has padded rows 3 and 4: unit diagonal in G
    expected[8, 8] += 1.0
    expected[9, 9] += 1.0
    size = 15
    full = np.zeros((size, size))
    for d in range(band.shape[0]):
        for c in range(size - d):
            full[c + d, c] = band[d, c]
            full[c, c + d] = band[d, c]
    np.testing.assert_allclose(full, expected, atol=1e-12)


def planar_hinge_scene(n):
    return build_chain(n, angles=np.linspace(1.0, 2.0, n), joint="hinge")


@pytest.mark.parametrize("n", [1, 3])
def test_hinge_chain_matches_lagrangian_reference(n):
    angles = np.linspace(1.0, 2.0, n)
    buffers = compile_scene(planar_hinge_scene(n))
    engine = Engine(buffers)
    steps = 1000
    engine.run(steps)
    ref = simulate_chain(angles, np.ones(n), np.ones(n), np.array([0.0, steps * buffers.dt]))[-1]
    y = np.array([2.0 * np.arctan2(buffers.q[i, 4], buffers.q[i, 6]) for i in range(n)])
    np.testing.assert_allclose(y, ref, atol=5e-3)


def test_hinge_keeps_axis_while_ball_does_not():
    scene = Scene()
    body = scene.add_body(Body(mass=1.0, inertia=[0.1, 0.1, 0.1], position=[0.5, 0, 0]))
    scene.add_hinge_joint(HingeJoint(WORLD, body, [0, 0, 0], [-0.5, 0, 0], [0, 1, 0], [0, 1, 0]))
    buffers = compile_scene(scene)
    buffers.u[0, 3:] = [1.0, 0.5, 2.0]  # spin about all axes
    engine = Engine(buffers)
    engine.run(2000)
    axis_world = quat.to_matrix(buffers.q[:, 3:])[0] @ np.array([0.0, 1.0, 0.0])
    assert np.linalg.norm(axis_world - [0, 1, 0]) < 1e-3
    assert (
        np.linalg.norm(buffers.q[0, :3] - quat.to_matrix(buffers.q[:, 3:])[0] @ [0.5, 0, 0]) < 1e-3
    )


def test_mixed_scene_runs_and_keeps_constraints():
    scene = Scene()
    a = scene.add_body(Body(mass=1.0, inertia=[0.1, 0.1, 0.1], position=[0.5, 0, 0]))
    b = scene.add_body(Body(mass=1.0, inertia=[0.1, 0.1, 0.1], position=[1.5, 0, 0]))
    scene.add_ball_joint(BallJoint(WORLD, a, [0, 0, 0], [-0.5, 0, 0]))
    scene.add_hinge_joint(HingeJoint(a, b, [0.5, 0, 0], [-0.5, 0, 0], [0, 1, 0], [0, 1, 0]))
    buffers = compile_scene(scene)
    engine = Engine(buffers)
    engine.run(1000)
    rot = quat.to_matrix(buffers.q[:, 3:])
    gap = (buffers.q[1, :3] + rot[1] @ [-0.5, 0, 0]) - (buffers.q[0, :3] + rot[0] @ [0.5, 0, 0])
    assert np.linalg.norm(gap) < 1e-3
    assert abs((rot[0] @ [0, 1, 0]) @ (rot[1] @ [0, 1, 0]) - 1.0) < 1e-3


def test_misaligned_hinge_axes_rejected():
    scene = Scene()
    body = scene.add_body(Body(mass=1.0, inertia=[0.1, 0.1, 0.1], position=[0.5, 0, 0]))
    scene.add_hinge_joint(HingeJoint(WORLD, body, [0, 0, 0], [-0.5, 0, 0], [0, 1, 0], [0, 0, 1]))
    with pytest.raises(ValueError, match="not aligned"):
        compile_scene(scene)
