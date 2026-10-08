"""Analytical ball joint Jacobian vs. finite differences of g(q)."""

import numpy as np
import pytest

from dynamix.numpy_backend import quaternion as quat
from dynamix.numpy_backend.constraints import ball_joint_jacobian, ball_joint_residual

EPS = 1e-6


def random_state(rng, n):
    q = np.empty((n, 7))
    q[:, :3] = rng.normal(size=(n, 3))
    quats = rng.normal(size=(n, 4))
    q[:, 3:] = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    return q


def perturb(q, du):
    """Apply a displacement du (n, 6): world-frame translation + body-frame rotation vector."""
    out = q.copy()
    out[:, :3] += du[:, :3]
    out[:, 3:] = quat.integrate(q[:, 3:], du[:, 3:], 1.0)
    return out


def numerical_jacobian(residual, q, eps=EPS):
    """Central differences of residual(q) w.r.t. the generalized displacement; shape (m, 6n)."""
    n = q.shape[0]
    cols = []
    for k in range(6 * n):
        du = np.zeros(6 * n)
        du[k] = eps
        plus = residual(perturb(q, du.reshape(n, 6)))
        du[k] = -eps
        minus = residual(perturb(q, du.reshape(n, 6)))
        cols.append((plus - minus).reshape(-1) / (2 * eps))
    return np.stack(cols, axis=1)


def make_case(rng, kind):
    anchors = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    if kind == "world":
        q = random_state(rng, 2)
        a, b = np.array([-1, -1, -1]), np.array([0, 1, 0])
    else:
        q = random_state(rng, 3)
        a, b = np.array([0, 1, 2]), np.array([1, 2, 0])
    return q, a, b, *anchors


@pytest.mark.parametrize("kind", ["world", "body"])
@pytest.mark.parametrize("seed", range(5))
def test_jacobian_matches_finite_differences(kind, seed):
    q, a, b, ra, rb = make_case(np.random.default_rng(seed), kind)

    def residual(state):
        return ball_joint_residual(state, a, b, ra, rb)

    analytical = ball_joint_jacobian(q, a, b, ra, rb)
    numerical = numerical_jacobian(residual, q)
    assert analytical.shape == (9, 6 * q.shape[0])
    np.testing.assert_allclose(analytical, numerical, atol=1e-7)


def test_jacobian_consistent_with_integrator():
    rng = np.random.default_rng(42)
    q, a, b, ra, rb = make_case(rng, "body")
    u = rng.normal(size=(q.shape[0], 6))
    dt = 1e-6
    q_next = q.copy()
    q_next[:, :3] += dt * u[:, :3]
    q_next[:, 3:] = quat.integrate(q[:, 3:], u[:, 3:], dt)
    g_dot = (ball_joint_residual(q_next, a, b, ra, rb) - ball_joint_residual(q, a, b, ra, rb)) / dt
    jac_u = ball_joint_jacobian(q, a, b, ra, rb) @ u.reshape(-1)
    np.testing.assert_allclose(jac_u, g_dot.reshape(-1), atol=1e-4)


def test_out_buffer_is_reused_and_overwritten():
    q, a, b, ra, rb = make_case(np.random.default_rng(1), "world")
    out = np.full((9, 12), 7.0)
    result = ball_joint_jacobian(q, a, b, ra, rb, out=out)
    assert result is out
    np.testing.assert_allclose(out, ball_joint_jacobian(q, a, b, ra, rb))
