import numpy as np

from dynamix.numpy_backend import quaternion as quat

RNG = np.random.default_rng(0)


def random_quats(n):
    q = RNG.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_matrix_is_rotation():
    r = quat.to_matrix(random_quats(5))
    np.testing.assert_allclose(
        r @ r.transpose(0, 2, 1), np.broadcast_to(np.eye(3), r.shape), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-12)


def test_right_handed_z_rotation():
    q = quat.from_rotvec(np.array([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(quat.to_matrix(q) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_multiply_matches_matrix_product():
    a, b = random_quats(4), random_quats(4)
    np.testing.assert_allclose(
        quat.to_matrix(quat.multiply(a, b)), quat.to_matrix(a) @ quat.to_matrix(b), atol=1e-12
    )


def test_from_rotvec_small_angle_continuous():
    q = quat.from_rotvec(np.array([[1e-9, 0, 0], [1e-7, 0, 0]]))
    np.testing.assert_allclose(q[:, 0], [0.5e-9, 0.5e-7], rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)


def test_integrate_is_body_frame_rotation():
    q = random_quats(1)
    w = np.array([[0.3, -0.2, 0.5]])
    qn = quat.integrate(q, w, 0.1)
    expected = quat.to_matrix(q) @ quat.to_matrix(quat.from_rotvec(w * 0.1))
    np.testing.assert_allclose(quat.to_matrix(qn), expected, atol=1e-12)


def test_skew_is_cross_product():
    v, x = RNG.normal(size=(2, 3))
    np.testing.assert_allclose(quat.skew(v) @ x, np.cross(v, x), atol=1e-14)


def test_integrate_angular_velocity_along_body_axis_keeps_axis_fixed():
    # rotating about the body x-axis leaves that axis (in world coordinates) unchanged
    q = random_quats(1)
    axis_world = quat.to_matrix(q)[0] @ [1.0, 0.0, 0.0]
    qn = quat.integrate(q, np.array([[2.0, 0.0, 0.0]]), 0.3)
    np.testing.assert_allclose(quat.to_matrix(qn)[0] @ [1.0, 0.0, 0.0], axis_world, atol=1e-12)
