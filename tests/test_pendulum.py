import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dynamix.core import compile_scene
from dynamix.ecs import BallJoint, Body, Gravity, Scene, SimulationConfig
from dynamix.numpy_backend import Engine, total_energy
from dynamix.numpy_backend import quaternion as quat
from dynamix.numpy_backend.constraints import ball_joint_residual
from dynamix.scenes import build_pendulum

G, LENGTH, MASS = 9.81, 1.0, 1.0


def pendulum_angle(buf):
    """Rotation angle about +y recovered from the quaternion."""
    return 2.0 * np.arctan2(buf.q[0, 4], buf.q[0, 6])


def reference_angle(angle0, t_end, n):
    inertia_pivot = MASS * LENGTH**2 / 12.0 + MASS * (LENGTH / 2) ** 2
    k = MASS * G * LENGTH / 2 / inertia_pivot
    t = np.linspace(0.0, t_end, n)
    sol = solve_ivp(
        lambda _, y: [y[1], -k * np.sin(y[0])],
        (0, t_end),
        [angle0, 0.0],
        t_eval=t,
        rtol=1e-11,
        atol=1e-12,
    )
    return t, sol.y[0]


def run(angle0, dt, t_end, **cfg):
    scene = build_pendulum(
        angle=angle0, length=LENGTH, mass=MASS, config=SimulationConfig(dt=dt, **cfg)
    )
    buf = compile_scene(scene)
    engine = Engine(buf)
    steps = round(t_end / dt)
    angles = [pendulum_angle(buf)]
    for _ in range(steps):
        engine.step()
        angles.append(pendulum_angle(buf))
    return buf, np.array(angles)


@pytest.mark.parametrize("angle0", [0.1, 1.0])
def test_matches_reference_ode(angle0):
    dt, t_end = 1e-3, 3.0
    buf, angles = run(angle0, dt, t_end)
    _, ref = reference_angle(angle0, t_end, len(angles))
    np.testing.assert_allclose(angles, ref, atol=2e-3)


def test_small_angle_period():
    dt = 5e-4
    _, angles = run(0.02, dt, 4.0)
    inertia_pivot = MASS * LENGTH**2 / 3.0
    period = 2 * np.pi * np.sqrt(inertia_pivot / (MASS * G * LENGTH / 2))
    # first downward zero crossing after t=0 is at a quarter period
    idx = np.argmax(angles < 0.0)
    t_cross = (idx - angles[idx] / (angles[idx] - angles[idx - 1])) * dt
    assert t_cross == pytest.approx(period / 4, rel=2e-3)


def test_energy_and_constraint_stay_bounded():
    e0 = total_energy(compile_scene(build_pendulum(angle=1.0)))
    buf, _ = run(1.0, 1e-3, 1.0)
    assert abs(total_energy(buf) - e0) / abs(e0) < 1e-3
    g = ball_joint_residual(buf.q, buf.joint_a, buf.joint_b, buf.joint_anchor_a, buf.joint_anchor_b)
    assert np.linalg.norm(g) < 1e-4


def test_quaternion_stays_unit_and_motion_planar():
    buf, _ = run(1.0, 1e-3, 1.0)
    assert np.linalg.norm(buf.q[0, 3:]) == pytest.approx(1.0, abs=1e-12)
    assert abs(buf.q[0, 1]) < 1e-9


def test_compile_scene_buffers():
    scene = Scene()
    a = scene.add_body(
        Body(mass=2.0, inertia=[1, 2, 3], position=[1, 2, 3], orientation=[0, 0, 0, 2.0])
    )
    b = scene.add_body(Body(mass=1.0, inertia=np.eye(3)))
    scene.add_ball_joint(BallJoint(a, b, [0, 0, 1], [0, 0, -1]))
    buf = compile_scene(scene)
    assert buf.q.shape == (2, 7) and buf.u.shape == (2, 6)
    np.testing.assert_allclose(buf.q[0], [1, 2, 3, 0, 0, 0, 1])
    np.testing.assert_allclose(buf.inertia_body[0], np.diag([1, 2, 3]))
    assert buf.joint_a.tolist() == [0] and buf.joint_b.tolist() == [1]
    assert buf.q.flags.c_contiguous and buf.body_entities == (a, b)


def test_two_body_chain_keeps_joints_closed():
    scene = Scene(config=SimulationConfig(dt=1e-3))
    a = scene.add_body(Body(mass=1.0, inertia=[0.1, 0.1, 0.1], position=[0.5, 0, 0]))
    b = scene.add_body(Body(mass=1.0, inertia=[0.1, 0.1, 0.1], position=[1.5, 0, 0]))
    scene.add_ball_joint(BallJoint(-1, a, [0, 0, 0], [-0.5, 0, 0]))
    scene.add_ball_joint(BallJoint(a, b, [0.5, 0, 0], [-0.5, 0, 0]))
    buf = compile_scene(scene)
    engine = Engine(buf)
    e0 = total_energy(buf)
    engine.run(1000)
    g = ball_joint_residual(buf.q, buf.joint_a, buf.joint_b, buf.joint_anchor_a, buf.joint_anchor_b)
    assert np.linalg.norm(g) < 1e-3
    # energy scale: m g L of the two-link chain
    assert abs(total_energy(buf) - e0) < 1e-2 * 2 * 9.81 * 2.0


def test_torque_free_body_conserves_angular_momentum_and_energy():
    # Asymmetric top (Euler equations in the body frame): |L| and kinetic energy are conserved.
    scene = Scene(config=SimulationConfig(dt=1e-4), gravity=Gravity([0.0, 0.0, 0.0]))
    scene.add_body(Body(mass=1.0, inertia=[1.0, 2.0, 3.0], angular_velocity=[0.5, 2.0, 0.3]))
    buf = compile_scene(scene)
    engine = Engine(buf)

    def momentum():
        rot = quat.to_matrix(buf.q[0, 3:])
        return rot @ (buf.inertia_body[0] @ buf.u[0, 3:])

    l0, e0 = momentum(), total_energy(buf)
    engine.run(20000)
    np.testing.assert_allclose(momentum(), l0, rtol=0, atol=1e-3 * np.linalg.norm(l0))
    assert total_energy(buf) == pytest.approx(e0, rel=1e-3)
