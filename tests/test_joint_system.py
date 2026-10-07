"""Block-sparse ball joint system vs. the dense reference formulation."""

import numpy as np
import pytest

from dynamix.numpy_backend import banded
from dynamix.numpy_backend import quaternion as quat
from dynamix.numpy_backend.banded import banded_to_dense, solve_spd_banded
from dynamix.numpy_backend.constraints import ball_joint_jacobian, ball_joint_residual
from dynamix.numpy_backend.joint_system import JointSystem

# (joint_a, joint_b) lists: chain, branched tree with loop, joints out of body order
TOPOLOGIES = {
    "chain": ([-1, 0, 1, 2], [0, 1, 2, 3]),
    "branched_with_loop": ([-1, 0, 1, 1, 3, 0], [0, 1, 2, 3, 4, 3]),
    "unordered": ([2, -1, 0, 1], [3, 0, 1, 2]),
}


def make_system(name, seed=0):
    rng = np.random.default_rng(seed)
    joint_a, joint_b = (np.array(x) for x in TOPOLOGIES[name])
    n = int(max(joint_a.max(), joint_b.max())) + 1
    q = np.empty((n, 7))
    q[:, :3] = rng.normal(size=(n, 3))
    quats = rng.normal(size=(n, 4))
    q[:, 3:] = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    mass = rng.uniform(0.5, 2.0, n)
    inertia = np.array([np.diag(rng.uniform(0.1, 1.0, 3)) for _ in range(n)])
    anchor_a, anchor_b = rng.normal(size=(2, len(joint_a), 3))
    system = JointSystem(joint_a, joint_b, anchor_a, anchor_b, mass, np.linalg.inv(inertia))
    rot = quat.to_matrix(q[:, 3:])
    system.update(q, rot)
    return system, q, rot, (joint_a, joint_b, anchor_a, anchor_b), mass, inertia


def dense_jacobian(system, n_bodies):
    jac = np.zeros((3 * system.n_joints, 6 * n_bodies))
    for s in range(system.n_sides):
        rows = slice(3 * system.side_joint[s], 3 * system.side_joint[s] + 3)
        body = system.side_body[s]
        jac[rows, 6 * body : 6 * body + 3] += system.sign[s] * np.eye(3)
        jac[rows, 6 * body + 3 : 6 * body + 6] += system.jw[s]
    return jac


def dense_inverse_mass(mass, inertia):
    n = len(mass)
    minv = np.zeros((6 * n, 6 * n))
    for i in range(n):
        minv[6 * i : 6 * i + 3, 6 * i : 6 * i + 3] = np.eye(3) / mass[i]
        minv[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = np.linalg.inv(inertia[i])
    return minv


@pytest.mark.parametrize("name", TOPOLOGIES)
def test_blocks_and_residual_match_dense_reference(name):
    system, q, rot, (ja, jb, ra, rb), _, _ = make_system(name)
    dense = ball_joint_jacobian(q, ja, jb, ra, rb)
    np.testing.assert_allclose(dense_jacobian(system, q.shape[0]), dense, atol=1e-12)
    np.testing.assert_allclose(system.g, ball_joint_residual(q, ja, jb, ra, rb), atol=1e-12)


@pytest.mark.parametrize("name", TOPOLOGIES)
def test_banded_delassus_matches_dense_product(name):
    system, q, _, _, mass, inertia = make_system(name)
    jac = dense_jacobian(system, q.shape[0])
    expected = jac @ dense_inverse_mass(mass, inertia) @ jac.T
    band = system.delassus_band()
    assert band.shape == (system.half_bandwidth + 1, 3 * system.n_joints)
    np.testing.assert_allclose(banded_to_dense(band), expected, atol=1e-12)


def test_chain_has_constant_bandwidth():
    n = 30
    joint_a = np.arange(-1, n - 1)
    joint_b = np.arange(n)
    anchors = np.zeros((n, 3))
    system = JointSystem(
        joint_a, joint_b, anchors, anchors, np.ones(n), np.broadcast_to(np.eye(3), (n, 3, 3))
    )
    assert system.half_bandwidth == 5


@pytest.mark.parametrize("name", TOPOLOGIES)
def test_velocity_product_and_impulse_match_dense(name):
    rng = np.random.default_rng(3)
    system, q, _, _, mass, inertia = make_system(name)
    n = q.shape[0]
    jac = dense_jacobian(system, n)
    u = rng.normal(size=(n, 6))
    np.testing.assert_allclose(system.jacobian_times(u), jac @ u.reshape(-1), atol=1e-12)

    impulse = rng.normal(size=(system.n_joints, 3))
    system.delassus_band()  # fills the M^-1 J^T work blocks
    expected = u.reshape(-1) + dense_inverse_mass(mass, inertia) @ jac.T @ impulse.reshape(-1)
    system.add_impulse(impulse, u)
    np.testing.assert_allclose(u.reshape(-1), expected, atol=1e-12)


def test_banded_solve_matches_dense_and_fallback(monkeypatch):
    system, q, _, _, mass, inertia = make_system("chain")
    band = system.delassus_band()
    rhs = np.random.default_rng(1).normal(size=3 * system.n_joints)
    expected = np.linalg.solve(banded_to_dense(band), rhs)
    np.testing.assert_allclose(solve_spd_banded(band, rhs), expected, atol=1e-9)
    monkeypatch.setattr(banded, "solveh_banded", None)
    np.testing.assert_allclose(banded.solve_spd_banded(band, rhs), expected, atol=1e-9)


def test_rejects_degenerate_joints():
    eye = np.broadcast_to(np.eye(3), (2, 3, 3))
    zeros = np.zeros((1, 3))
    with pytest.raises(ValueError):
        JointSystem([1], [1], zeros, zeros, np.ones(2), eye)
    with pytest.raises(ValueError):
        JointSystem([], [], np.zeros((0, 3)), np.zeros((0, 3)), np.ones(2), eye)
