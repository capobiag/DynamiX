import numpy as np
import pytest

from dynamix.core import compile_scene
from dynamix.ecs import SimulationConfig
from dynamix.numpy_backend import Engine, total_energy
from dynamix.numpy_backend.constraints import ball_joint_residual
from dynamix.scenes import build_chain
from reference import simulate_chain


def link_angles(buf):
    """Absolute rotation angle of each link about +y."""
    return 2.0 * np.arctan2(buf.q[:, 4], buf.q[:, 6])


def joint_gap(buf):
    return ball_joint_residual(
        buf.q, buf.joint_a, buf.joint_b, buf.joint_anchor_a, buf.joint_anchor_b
    )


def simulate(n, phi0, t_end, dt=5e-4, **kwargs):
    buf = compile_scene(build_chain(n, phi0, config=SimulationConfig(dt=dt), **kwargs))
    engine = Engine(buf)
    steps = round(t_end / dt)
    angles = [link_angles(buf)]
    for _ in range(steps):
        engine.step()
        angles.append(link_angles(buf))
    return buf, np.array(angles), np.arange(steps + 1) * dt


def test_build_chain_topology_and_initial_closure():
    buf = compile_scene(build_chain(4, [0.3, -0.4, 1.0, 2.0], length=[1, 0.5, 0.7, 0.2]))
    assert buf.n_bodies == 4 and buf.n_joints == 4
    assert buf.joint_a.tolist() == [-1, 0, 1, 2] and buf.joint_b.tolist() == [0, 1, 2, 3]
    assert np.abs(joint_gap(buf)).max() < 1e-14


def test_build_chain_validates_arguments():
    with pytest.raises(ValueError):
        build_chain(0)
    with pytest.raises(ValueError):
        build_chain(3, length=[1.0, 2.0])


@pytest.mark.parametrize("n, t_end, tol", [(2, 1.5, 5e-3), (3, 1.0, 5e-3)])
def test_matches_lagrangian_reference(n, t_end, tol):
    phi0 = np.linspace(1.0, 0.3, n)
    lengths = np.linspace(1.0, 0.6, n)
    masses = np.linspace(1.0, 2.0, n)
    buf, angles, t = simulate(n, phi0, t_end, length=lengths, mass=masses)
    ref = simulate_chain(phi0, lengths, masses, t)
    np.testing.assert_allclose(angles, ref, atol=tol)


def test_converges_with_time_step():
    phi0, n, t_end = [1.0, 0.5], 2, 1.0
    ref = simulate_chain(phi0, [1.0] * n, [1.0] * n, np.array([0.0, t_end]))[-1]
    errors = [
        np.abs(simulate(n, phi0, t_end, dt=dt)[1][-1] - ref).max() for dt in (2e-3, 1e-3, 5e-4)
    ]
    assert errors[1] < 0.5 * errors[0] and errors[2] < 0.5 * errors[1]


@pytest.mark.parametrize("n", [2, 5, 10])
def test_energy_constraints_and_planarity(n):
    buf = compile_scene(build_chain(n, config=SimulationConfig(dt=5e-4)))
    engine = Engine(buf)
    e0 = total_energy(buf)
    scale = float(buf.mass.sum() * 9.81 * n)
    worst_gap = 0.0
    for _ in range(2000):
        engine.step()
        worst_gap = max(worst_gap, np.abs(joint_gap(buf)).max())
    assert abs(total_energy(buf) - e0) < 2e-3 * scale
    assert worst_gap < 1e-3
    assert np.abs(buf.q[:, 1]).max() < 1e-8  # no out-of-plane drift
    np.testing.assert_allclose(np.linalg.norm(buf.q[:, 3:], axis=1), 1.0, atol=1e-12)
