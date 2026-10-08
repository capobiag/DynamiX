"""Moreau theta-method time stepping with velocity-level bilateral constraints (JAX).

Same step as ``dynamix.numpy_backend.engine`` (see docs/theory.md):
    q_th   = q + theta dt u                      (midpoint position; q_rot * exp(theta dt w))
    M(u+ - u) = dt h(u) + J(q_th)^T lam
    J(q_th) u+ = -(stabilization / dt) g(q_th)
    q+     = q_th + (1 - theta) dt u+
Normal contacts (``contact_cfg`` given) are detected at q_th and solved after the joints; see
``contacts.py``. ``step`` is a pure function of ``(state, params, system)``; ``run`` and
``rollout`` fuse many steps into a single compiled ``lax.scan``. All functions are vmap-able over
states; without contacts they are also differentiable with ``jax.grad`` (the contact iteration
is a ``while_loop``).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import cho_solve

from dynamix.core import joint_math as jm
from dynamix.core.buffers import SystemBuffers
from dynamix.jax_backend import contacts as bc
from dynamix.jax_backend import joint_system as bjs
from dynamix.jax_backend import quaternion as quat
from dynamix.jax_backend.banded import (
    block_band_to_dense,
    cholesky_block_banded,
    solve_cholesky_block_banded,
    solve_spd_block_banded,
)

# Block counts up to which a dense Cholesky of G is faster than the block-banded scan.
DENSE_SOLVE_MAX_JOINTS = 10


class SimulationState(NamedTuple):
    q: jnp.ndarray  # (n, 7) [x, y, z, qx, qy, qz, qw]
    u: jnp.ndarray  # (n, 6) [v, w], w in the body frame
    time: jnp.ndarray  # scalar


class SystemParams(NamedTuple):
    mass: jnp.ndarray  # (n,)
    inertia_body: jnp.ndarray  # (n, 3, 3)
    inertia_inv: jnp.ndarray  # (n, 3, 3)
    gravity: jnp.ndarray  # (3,)
    contacts: bc.ContactParams | None = None  # collider arrays; None without colliders


def make_state(buffers: SystemBuffers) -> SimulationState:
    return SimulationState(
        jnp.asarray(buffers.q), jnp.asarray(buffers.u), jnp.zeros((), dtype=buffers.q.dtype)
    )


def make_params(buffers: SystemBuffers):
    """Host-side setup: ``(params, system, block_bandwidth)``; ``system`` is None without joints."""
    inertia_inv = np.linalg.inv(buffers.inertia_body)
    params = SystemParams(
        jnp.asarray(buffers.mass),
        jnp.asarray(buffers.inertia_body),
        jnp.asarray(inertia_inv),
        jnp.asarray(buffers.gravity),
        bc.make_contact_params(buffers),
    )
    if buffers.n_joints == 0:
        return params, None, 0
    system, bandwidth = bjs.make_joint_system(
        buffers.joint_a,
        buffers.joint_b,
        buffers.joint_anchor_a,
        buffers.joint_anchor_b,
        buffers.mass,
        inertia_inv,
        buffers.joint_kind,
        buffers.joint_axis_a,
        buffers.joint_axis_b,
        buffers.joint_rel_rot,
    )
    return params, system, bandwidth


def _solve(band, rhs, dense):
    if dense:
        return jnp.linalg.solve(block_band_to_dense(band), rhs)
    return solve_spd_block_banded(band, rhs)


def _factor(band, dense):
    """Factor G once for the repeated joint solves of the contact iteration."""
    if dense:
        return jnp.linalg.cholesky(block_band_to_dense(band))
    return cholesky_block_banded(band)


def _solve_factored(factor, rhs, dense):
    if dense:
        return cho_solve((factor, True), rhs)
    return solve_cholesky_block_banded(factor, rhs)


@partial(
    jax.jit, static_argnames=("dt", "theta", "stabilization", "block_bandwidth", "contact_cfg")
)
def step(state, params, system, *, dt, theta, stabilization, block_bandwidth=0, contact_cfg=None):
    q, u = state.q, state.u
    v, w = u[:, :3], u[:, 3:]

    q_th = jnp.concatenate(
        (q[:, :3] + theta * dt * v, quat.integrate(q[:, 3:], w, theta * dt)), axis=1
    )
    rot = quat.to_matrix(q_th[:, 3:])

    gyro = jm.gyroscopic_acceleration(jnp, params.inertia_body, params.inertia_inv, w)
    u_free = jnp.concatenate((v + dt * params.gravity, w - dt * gyro), axis=1)

    if system is not None:
        g, jw, lin = bjs.update(system, q_th, rot)
        band, ww = bjs.delassus_band(system, jw, lin, block_bandwidth)
        rhs = -(stabilization / dt) * g - bjs.jacobian_times(system, jw, lin, u_free)
        dense = g.shape[0] <= DENSE_SOLVE_MAX_JOINTS
        if contact_cfg is None:
            impulse = _solve(band, rhs.reshape(-1), dense)
        else:
            factor = _factor(band, dense)
            impulse = _solve_factored(factor, rhs.reshape(-1), dense)
        u_free = bjs.add_impulse(system, lin, ww, impulse.reshape(g.shape), u_free)

    if contact_cfg is not None:
        contacts = bc.detect(params.contacts, contact_cfg, q_th)
        correct = None
        if system is not None:
            bias = -(stabilization / dt) * g

            def correct(x):
                u_x = x[:-1]
                rhs_x = (bias - bjs.jacobian_times(system, jw, lin, u_x)).reshape(-1)
                corr = _solve_factored(factor, rhs_x, dense).reshape(g.shape)
                return x.at[:-1].set(bjs.add_impulse(system, lin, ww, corr, u_x)), jnp.max(
                    jnp.abs(corr)
                )

        u_free = bc.solve(params.contacts, contact_cfg, dt, contacts, u, u_free, correct)

    v_new, w_new = u_free[:, :3], u_free[:, 3:]
    q_new = jnp.concatenate(
        (
            q_th[:, :3] + (1.0 - theta) * dt * v_new,
            quat.integrate(q_th[:, 3:], w_new, (1.0 - theta) * dt),
        ),
        axis=1,
    )
    return SimulationState(q_new, u_free, state.time + dt)


@partial(
    jax.jit,
    static_argnames=("steps", "dt", "theta", "stabilization", "block_bandwidth", "contact_cfg"),
)
def run(
    state,
    params,
    system,
    *,
    steps,
    dt,
    theta,
    stabilization,
    block_bandwidth=0,
    contact_cfg=None,
):
    """Advance ``steps`` steps inside one compiled loop."""
    kwargs = {"dt": dt, "theta": theta, "stabilization": stabilization, "contact_cfg": contact_cfg}

    def body(s, _):
        return _step_impl(s, params, system, block_bandwidth, **kwargs), None

    return lax.scan(body, state, None, length=steps)[0]


@partial(
    jax.jit,
    static_argnames=(
        "steps",
        "stride",
        "dt",
        "theta",
        "stabilization",
        "block_bandwidth",
        "contact_cfg",
    ),
)
def rollout(
    state,
    params,
    system,
    *,
    steps,
    stride=1,
    dt,
    theta,
    stabilization,
    block_bandwidth=0,
    contact_cfg=None,
):
    """Like ``run`` but also returns the states every ``stride`` steps."""
    kwargs = {"dt": dt, "theta": theta, "stabilization": stabilization, "contact_cfg": contact_cfg}

    def inner(s, _):
        return _step_impl(s, params, system, block_bandwidth, **kwargs), None

    def outer(s, _):
        s = lax.scan(inner, s, None, length=stride)[0]
        return s, s

    return lax.scan(outer, state, None, length=steps // stride)


def _step_impl(state, params, system, block_bandwidth, **kwargs):
    # ``step`` is jitted; calling it inside a trace inlines it.
    return step(state, params, system, block_bandwidth=block_bandwidth, **kwargs)


def total_energy(state, params):
    """Kinetic plus gravitational potential energy."""
    v, w = state.u[:, :3], state.u[:, 3:]
    kinetic = 0.5 * jnp.sum(params.mass * jnp.sum(v * v, axis=1))
    kinetic += 0.5 * jnp.einsum("ni,nij,nj->", w, params.inertia_body, w)
    potential = -jnp.sum(params.mass * (state.q[:, :3] @ params.gravity))
    return kinetic + potential


class Engine:
    """Convenience wrapper mirroring ``numpy_backend.Engine``; holds the (immutable) state."""

    def __init__(self, buffers: SystemBuffers, contact_slots: int = 4):
        self.params, self.system, self.block_bandwidth = make_params(buffers)
        self.contact_cfg = bc.make_contact_settings(buffers, contact_slots)
        self.state = make_state(buffers)
        self._kw = {
            "dt": float(buffers.dt),
            "theta": float(buffers.theta),
            "stabilization": float(buffers.stabilization),
            "block_bandwidth": self.block_bandwidth,
            "contact_cfg": self.contact_cfg,
        }

    @property
    def time(self) -> float:
        return float(self.state.time)

    def step(self) -> None:
        self.state = step(self.state, self.params, self.system, **self._kw)

    def run(self, steps: int) -> None:
        self.state = run(self.state, self.params, self.system, steps=steps, **self._kw)

    def rollout(self, steps: int, stride: int = 1):
        """Advance ``steps`` steps; return the recorded states every ``stride`` steps."""
        self.state, frames = rollout(
            self.state, self.params, self.system, steps=steps, stride=stride, **self._kw
        )
        return frames

    @property
    def contacts(self):
        """Contact buffer detected at the current configuration (None without colliders)."""
        if self.contact_cfg is None:
            return None
        return bc.detect(self.params.contacts, self.contact_cfg, self.state.q)

    def total_energy(self) -> float:
        return float(total_energy(self.state, self.params))
