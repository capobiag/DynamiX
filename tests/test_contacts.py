"""Normal contacts: SDF primitives, broad phase, solver, bouncing ball and joint coupling."""

import warnings

import numpy as np
import pytest

from dynamix.core import compile_scene
from dynamix.core import contacts as ct
from dynamix.ecs import (
    WORLD,
    BallJoint,
    Body,
    PlaneCollider,
    Scene,
    SimulationConfig,
    SphereCollider,
)
from dynamix.numpy_backend import Engine
from dynamix.numpy_backend.contacts import ContactDetector, all_pairs, grid_pairs
from dynamix.numpy_backend.engine import total_energy
from dynamix.scenes import build_ball_pile, build_bouncing_ball


def ball(scene, position, radius=0.5, mass=1.0, restitution=0.0, velocity=(0, 0, 0)):
    body = scene.add_body(
        Body(
            mass=mass,
            inertia=np.full(3, 0.4 * mass * radius**2),
            position=np.asarray(position, float),
            linear_velocity=np.asarray(velocity, float),
        )
    )
    scene.add_sphere_collider(body, SphereCollider(radius, restitution))
    return body


def zero_gravity_scene(config=None):
    from dynamix.ecs import Gravity

    return Scene(config, Gravity(np.zeros(3)))


# --- SDF primitives ---------------------------------------------------------------------------


def test_sphere_sdf_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    x, c = rng.normal(size=(2, 6, 3))
    r = rng.uniform(0.1, 0.5, 6)
    _, grad = ct.sphere_sdf(np, x, c, r)
    eps = 1e-6
    for k in range(3):
        dx = np.zeros(3)
        dx[k] = eps
        up, _ = ct.sphere_sdf(np, x + dx, c, r)
        dn, _ = ct.sphere_sdf(np, x - dx, c, r)
        np.testing.assert_allclose((up - dn) / (2 * eps), grad[:, k], atol=1e-7)


def test_plane_sdf_and_contact_point():
    normal = np.array([0.0, 0.6, 0.8])
    x = np.array([[0.3, 0.2, 0.5]])
    sdf, grad = ct.plane_sdf(np, x, normal, 0.1)
    gap, n, point = ct.sphere_vs_sdf(np, x, np.array([0.4]), sdf, grad)
    np.testing.assert_allclose(gap, normal @ x[0] - 0.1 - 0.4)
    # the midpoint of the two surfaces lies at distance gap/2 beyond the plane side of the sphere
    np.testing.assert_allclose(point[0], x[0] - (0.4 + 0.5 * gap[0]) * normal)


def test_sphere_contact_jacobian_matches_gap_derivative_and_has_no_angular_part():
    rng = np.random.default_rng(1)
    xa, xb = rng.normal(size=(2, 3))
    ra, rb = 0.5, 0.4
    xb = xa + 0.8 * np.array([0.0, 0.6, 0.8])

    def gap(xa, xb):
        sdf, grad = ct.sphere_sdf(np, xb[None], xa[None], np.array([ra]))
        return ct.sphere_vs_sdf(np, xb[None], rb, sdf, grad)

    g0, n, p = gap(xa, xb)
    eye = np.broadcast_to(np.eye(3), (1, 3, 3))
    ang_a, ang_b = ct.contact_arms(np, p, n, xa[None], xb[None], eye, eye)
    np.testing.assert_allclose(ang_a, 0.0, atol=1e-12)
    np.testing.assert_allclose(ang_b, 0.0, atol=1e-12)
    eps = 1e-6
    for k in range(3):
        dx = np.zeros(3)
        dx[k] = eps
        d_b = (gap(xa, xb + dx)[0] - gap(xa, xb - dx)[0]) / (2 * eps)
        d_a = (gap(xa + dx, xb)[0] - gap(xa - dx, xb)[0]) / (2 * eps)
        np.testing.assert_allclose(d_b, n[0, k], atol=1e-7)
        np.testing.assert_allclose(d_a, -n[0, k], atol=1e-7)


def test_contact_arms_general_matches_finite_difference():
    # off-centre contact point: the angular row is the derivative of (n . r) w.r.t. body rotation
    rng = np.random.default_rng(2)
    n = rng.normal(size=(1, 3))
    n /= np.linalg.norm(n)
    p, xb = rng.normal(size=(2, 1, 3))
    q = rng.normal(size=4)
    from dynamix.numpy_backend import quaternion as quat

    rot = quat.to_matrix((q / np.linalg.norm(q))[None])
    _, ang_b = ct.contact_arms(np, p, n, np.zeros((1, 3)), xb, rot, rot)
    np.testing.assert_allclose(ang_b[0], rot[0].T @ np.cross(p[0] - xb[0], n[0]))


# --- broad phase and buffer ---------------------------------------------------------------


def normalised(i, j):
    return set(zip(np.minimum(i, j).tolist(), np.maximum(i, j).tolist(), strict=True))


def test_grid_pairs_cover_all_close_pairs_exactly_once():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 4, size=(300, 3))
    cell = 0.6
    i, j = grid_pairs(x, cell)
    assert len(normalised(i, j)) == i.shape[0]  # no duplicates
    ai, aj = all_pairs(x.shape[0])
    close = np.linalg.norm(x[ai] - x[aj], axis=1) < cell
    assert normalised(ai[close], aj[close]) <= normalised(i, j)


def test_grid_detector_equals_brute_force():
    scene = build_ball_pile(150)
    buf = compile_scene(scene)
    q = buf.q.copy()
    q[:, :3] += np.random.default_rng(5).normal(scale=0.05, size=(q.shape[0], 3))
    out = []
    for grid in (True, False):
        c = ContactDetector(buf, use_grid=grid).detect(q)
        lo = np.minimum(c.body_a[: c.count], c.body_b[: c.count])
        hi = np.maximum(c.body_a[: c.count], c.body_b[: c.count])
        out.append(sorted(zip(lo, hi, np.round(c.gap[: c.count], 12), strict=True)))
        n = c.normal[: c.count]
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0)
    assert out[0] == out[1] and len(out[0]) > 0


def test_overflow_keeps_deepest_contacts_and_flags():
    scene = zero_gravity_scene(SimulationConfig(max_contacts=2))
    for k in range(4):
        ball(scene, [0.0, 0.0, 0.0 + 0.2 * k], radius=0.5)
    buf = compile_scene(scene)
    detector = ContactDetector(buf)
    with pytest.warns(UserWarning, match="overflow"):
        c = detector.detect(buf.q)
    assert c.overflow and c.count == 2
    # the deepest pair is the closest one (distance 0.2 -> gap -0.8)
    assert c.gap.min() == pytest.approx(-0.8)


def test_no_overflow_no_warning():
    buf = compile_scene(build_bouncing_ball(0.1, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = ContactDetector(buf).detect(buf.q)
    assert c.count == 1 and not c.overflow and c.body_a[0] == -1


# --- dynamics ---------------------------------------------------------------------------------


@pytest.mark.parametrize("e", [0.0, 0.5, 1.0])
def test_head_on_collision(e):
    scene = zero_gravity_scene()
    ball(scene, [-0.6, 0, 0], radius=0.5, mass=1.0, restitution=e, velocity=[1.0, 0, 0])
    ball(scene, [0.6, 0, 0], radius=0.5, mass=3.0, restitution=e, velocity=[-0.5, 0, 0])
    buf = compile_scene(scene)
    engine = Engine(buf)
    p0 = buf.mass @ buf.u[:, 0]
    approach = 1.5
    engine.run(400)
    assert buf.mass @ buf.u[:, 0] == pytest.approx(p0, abs=1e-9)
    assert buf.u[1, 0] - buf.u[0, 0] == pytest.approx(e * approach, abs=1e-6)


def test_bouncing_ball_heights_and_energy():
    h, r, e = 1.0, 0.1, 0.8
    buf = compile_scene(build_bouncing_ball(h, r, 1.0, e))
    engine = Engine(buf)
    z = []
    energy = []
    for _ in range(3000):
        engine.step()
        z.append(buf.q[0, 2])
        energy.append(total_energy(buf))
    z = np.array(z)
    peaks = [z[i] for i in range(1, len(z) - 1) if z[i] > z[i - 1] and z[i] >= z[i + 1]]
    for k in range(3):
        expected = r + (h - r) * e ** (2 * (k + 1))
        assert peaks[k] == pytest.approx(expected, abs=0.01)
    assert z.min() > r - 0.01
    # restitution < 1 never creates energy (up to the O(dt) integration error)
    assert np.max(np.diff(energy)) < 1e-2


def test_inelastic_ball_comes_to_rest_on_the_plane():
    config = SimulationConfig(contact_stabilization=0.2)
    buf = compile_scene(build_bouncing_ball(0.5, 0.1, 1.0, 0.0, config))
    engine = Engine(buf)
    engine.run(2000)
    assert buf.q[0, 2] == pytest.approx(0.1, abs=1e-3)
    assert abs(buf.u[0, 2]) < 1e-3


def test_spin_is_preserved_without_friction():
    scene = build_bouncing_ball(0.3, 0.1, 1.0, 0.5)
    buf = compile_scene(scene)
    buf.u[0, 3:] = [1.0, 2.0, 3.0]
    Engine(buf).run(800)
    np.testing.assert_allclose(buf.u[0, 3:], [1.0, 2.0, 3.0], atol=1e-9)


def test_solution_satisfies_complementarity():
    buf = compile_scene(build_ball_pile(60))
    engine = Engine(buf)
    engine.run(300)
    c = engine.contacts
    assert c.count > 0
    n = c.normal[: c.count]
    a, b = c.body_a[: c.count], c.body_b[: c.count]
    u = np.vstack([buf.u[:, :3], np.zeros(3)])
    # post-step velocities, approximately: no contact is still approaching
    rel = np.einsum("ci,ci->c", n, u[b] - u[a])
    assert rel.min() > -5e-3


def sequential_pgs(contacts, u, inv_mass, rhs, sweeps=500):
    """Gauss-Seidel reference for the same frictionless LCP."""
    u = np.vstack([u, np.zeros(3)]).copy()
    c = contacts.count
    a = np.where(contacts.body_a[:c] < 0, u.shape[0] - 1, contacts.body_a[:c])
    b = contacts.body_b[:c]
    n = contacts.normal[:c]
    lam = np.zeros(c)
    for _ in range(sweeps):
        for k in range(c):
            w = inv_mass[a[k]] + inv_mass[b[k]]
            rel = n[k] @ (u[b[k]] - u[a[k]])
            new = max(0.0, lam[k] - (rel - rhs[k]) / w)
            d = new - lam[k]
            lam[k] = new
            u[b[k]] += d * inv_mass[b[k]] * n[k]
            u[a[k]] -= d * inv_mass[a[k]] * n[k]
    return u[:-1]


def test_jacobi_solver_matches_sequential_gauss_seidel():
    scene = zero_gravity_scene(SimulationConfig(contact_iterations=2000, contact_tolerance=1e-13))
    rng = np.random.default_rng(7)
    for k in range(5):
        ball(scene, [0.8 * k, 0.0, 0.0] + rng.normal(scale=0.05, size=3), 0.5, rng.uniform(1, 3))
    scene.add_plane(PlaneCollider(normal=np.array([0.0, 0.0, 1.0]), offset=-0.45))
    buf = compile_scene(scene)
    buf.u[:, :3] = rng.normal(size=(5, 3))
    u0 = buf.u.copy()
    engine = Engine(buf)
    # one step with dt tiny so that the geometry is frozen
    engine.buf.dt = 1e-6
    engine.step()
    c = engine.contacts
    inv_mass = np.append(1.0 / buf.mass, 0.0)
    ref = sequential_pgs(c, u0[:, :3], inv_mass, np.zeros(c.count))
    np.testing.assert_allclose(buf.u[:, :3], ref, atol=1e-6)


# --- joints and contacts ----------------------------------------------------------------------


def test_pendulum_bob_hitting_plane_keeps_joint_closed():
    from dynamix.ecs import Gravity

    scene = Scene(SimulationConfig(), Gravity(np.array([0.0, 0.0, -9.81])))
    length, r = 1.0, 0.1
    bob = ball(scene, [length, 0.0, 0.0], radius=r, restitution=0.5)
    scene.add_ball_joint(BallJoint(WORLD, bob, np.zeros(3), np.array([-length, 0.0, 0.0])))
    scene.add_plane(PlaneCollider(offset=-length + 0.1, restitution=0.5))
    buf = compile_scene(scene)
    engine = Engine(buf)
    hit = False
    worst_gap = worst_pen = 0.0
    for _ in range(3000):
        engine.step()
        worst_gap = max(worst_gap, abs(np.linalg.norm(buf.q[0, :3]) - length))
        worst_pen = min(worst_pen, buf.q[0, 2] - r - (-length + 0.1))
        hit |= engine.contacts is not None and engine.contacts.count > 0
    assert hit
    assert worst_gap < 1e-3
    assert worst_pen > -0.01


def test_scenes_without_colliders_are_unchanged():
    buf = compile_scene(build_bouncing_ball(1.0, 0.1))
    assert buf.has_contacts
    from dynamix.scenes import build_pendulum

    assert not compile_scene(build_pendulum()).has_contacts


def test_collider_validation():
    with pytest.raises(ValueError):
        SphereCollider(0.0)
    with pytest.raises(ValueError):
        SphereCollider(1.0, restitution=1.5)
    with pytest.raises(ValueError):
        PlaneCollider(normal=np.zeros(3))


def test_ball_pile_is_a_non_penetrating_cube_above_a_tilted_plane():
    scene = build_ball_pile(64, radius=0.1, tilt=0.4)
    buf = compile_scene(scene)
    x = buf.q[:, :3]
    dist = np.linalg.norm(x[:, None] - x[None], axis=-1) + 10.0 * np.eye(64)
    assert dist.min() >= 0.2 - 1e-12  # no overlaps, cubic spacing
    assert len(np.unique(np.round(x[:, 0], 9))) == 4  # 4 x 4 x 4 lattice, aligned with the axes
    normal = buf.plane_normal[0]
    assert abs(normal[0]) > 0.1 and normal[1] == 0.0  # plane tilted against the cube
    clearance = x @ normal - buf.plane_offset[0] - 0.1
    assert clearance.min() > 0.0
    assert ContactDetector(buf).detect(buf.q).count == 0


def test_sphere_touching_two_planes_gets_one_contact_per_plane():
    scene = zero_gravity_scene()
    ball(scene, [0.1, 0.0, 0.1], radius=0.2)
    scene.add_plane(PlaneCollider(normal=np.array([0.0, 0.0, 1.0]), offset=0.0))
    scene.add_plane(PlaneCollider(normal=np.array([1.0, 0.0, 0.0]), offset=0.0))
    buf = compile_scene(scene)
    c = ContactDetector(buf).detect(buf.q)
    assert c.count == 2
    np.testing.assert_allclose(sorted(c.normal[:2].tolist()), [[0, 0, 1], [1, 0, 0]])
    np.testing.assert_allclose(sorted(c.gap[:2]), [-0.3, -0.1])
