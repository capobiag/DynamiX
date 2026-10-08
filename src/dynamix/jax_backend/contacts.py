"""Normal contacts (JAX): sort-based grid detection and a block-iteration solver.

Same formulation as ``dynamix.numpy_backend.contacts`` and ``contact_solver`` (see docs/theory.md,
"Contacts"), restructured for static shapes under ``jit``:

- *Detection* sorts the spheres by grid cell (cell edge = largest diameter) and, for every sphere,
  scans the next ``slots`` spheres of its own cell and of 13 forward neighbour cells. That is
  O(n log n + 14 slots n) with fixed shapes. The hits are compacted into the fixed-capacity
  ``ContactBuffer`` by prefix sum and scatter; if there are more hits than the
  capacity, the deepest ``capacity`` contacts are kept (``lax.top_k``, evaluated only then).
  ``overflow`` is set when contacts were dropped, either because of the capacity or because a
  grid cell held more than ``slots`` spheres.
- *Solving* is the projected Jacobi iteration of the NumPy backend in a ``lax.while_loop`` with
  the same early exit. Without joints only the (n + 1, 3) linear velocities are iterated; with
  joints the full velocity is, and each sweep is followed by a joint correction that reuses the
  Cholesky factor of the joint Delassus matrix.
  The world is body index ``n`` of the extended arrays (zero inverse mass); entries of the buffer
  beyond ``count`` get zero weight. The loop is not reverse-mode differentiable.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

from dynamix.core import contacts as ct
from dynamix.core.buffers import SystemBuffers
from dynamix.core.contacts import ContactBuffer

_OFFSETS = np.array([(0, 0, 0), *ct.FORWARD_CELLS], dtype=np.int64)


class ContactSettings(NamedTuple):
    """Static (hashable) contact configuration; passed to ``jit`` as a static argument."""

    capacity: int  # contact buffer size
    slots: int  # spheres scanned per neighbour cell; more in one cell sets ``overflow``
    iterations: int
    tolerance: float
    omega: float
    stabilization: float  # penetration recovery gain beta
    restitution_threshold: float
    cell: float  # grid cell edge, the largest sphere diameter


class ContactParams(NamedTuple):
    """Collider arrays (PyTree leaves)."""

    sphere: jnp.ndarray  # (ns,) body index of each sphere collider
    radius: jnp.ndarray  # (ns,)
    restitution: jnp.ndarray  # (ns,)
    plane_normal: jnp.ndarray  # (np, 3)
    plane_offset: jnp.ndarray  # (np,)
    plane_restitution: jnp.ndarray  # (np,)
    inv_mass: jnp.ndarray  # (n + 1,) inverse masses with a trailing 0 for the world


def make_contact_params(buffers: SystemBuffers) -> ContactParams | None:
    if not buffers.has_contacts:
        return None
    spheres = np.flatnonzero(buffers.collider_radius > 0.0)
    return ContactParams(
        jnp.asarray(spheres),
        jnp.asarray(buffers.collider_radius[spheres]),
        jnp.asarray(buffers.collider_restitution[spheres]),
        jnp.asarray(buffers.plane_normal),
        jnp.asarray(buffers.plane_offset),
        jnp.asarray(buffers.plane_restitution),
        jnp.asarray(np.append(1.0 / buffers.mass, 0.0)),
    )


def make_contact_settings(buffers: SystemBuffers, slots: int = 4) -> ContactSettings | None:
    if not buffers.has_contacts:
        return None
    return ContactSettings(
        int(buffers.max_contacts),
        int(slots),
        int(buffers.contact_iterations),
        float(buffers.contact_tolerance),
        float(buffers.contact_omega),
        float(buffers.contact_stabilization),
        float(buffers.restitution_threshold),
        2.0 * float(buffers.collider_radius.max()),
    )


def _pair_candidates(cp: ContactParams, cfg: ContactSettings, x):
    """Cheap broad+narrow test of all grid candidates; geometry is evaluated later, on hits only.

    Returns the flat hit mask, sorted positions ``(pos_a, pos_b)`` of every candidate pair, the
    sort ``order`` and the cell overflow flag.
    """
    ns, slots = x.shape[0], cfg.slots
    index = jnp.floor((x - x.min(axis=0)) / cfg.cell).astype(jnp.int64) + 1  # >= 1 for -1 offsets
    dims = index.max(axis=0) + 2
    key = (index[:, 0] * dims[1] + index[:, 1]) * dims[2] + index[:, 2]
    order = jnp.argsort(key)
    skey, xs, radius = key[order], x[order], cp.radius[order]

    offsets = jnp.asarray(_OFFSETS)
    target = skey[:, None] + (offsets[:, 0] * dims[1] + offsets[:, 1]) * dims[2] + offsets[:, 2]
    start = jnp.searchsorted(skey, target.reshape(-1)).reshape(target.shape)
    start = start.at[:, 0].set(jnp.arange(1, ns + 1, dtype=start.dtype))  # own cell: later spheres
    pos = start[:, :, None] + jnp.arange(slots, dtype=start.dtype)  # (ns, 14, slots)
    safe = jnp.minimum(pos, ns - 1)
    valid = (pos < ns) & (skey[safe] == target[:, :, None])

    positions = jnp.arange(ns)
    reach = jnp.minimum(positions + slots, ns - 1)
    cell_overflow = jnp.any((positions + slots < ns) & (skey[reach] == skey))

    delta = xs[safe] - xs[:, None, None, :]
    reach_sum = radius[safe] + radius[:, None, None]
    hit = valid & (jnp.sum(delta * delta, axis=-1) <= reach_sum * reach_sum)
    pos_a = jnp.broadcast_to(positions[:, None, None], hit.shape)
    return hit.reshape(-1), pos_a.reshape(-1), safe.reshape(-1), order, cell_overflow


def detect(cp: ContactParams, cfg: ContactSettings, q) -> ContactBuffer:
    """Contacts with ``gap <= 0`` at the configuration ``q`` as a fixed-capacity buffer.

    The cheap overlap test runs on every candidate (sphere pairs from the grid, every sphere
    against every plane); the SDF geometry only on the ``capacity`` selected hits.
    """
    x = q[cp.sphere, :3]
    ns, n_planes = x.shape[0], cp.plane_offset.shape[0]
    hits, cell_overflow = [], jnp.asarray(False)
    n_pairs = 0
    if ns >= 2:
        pair_hit, pos_a, pos_b, order, cell_overflow = _pair_candidates(cp, cfg, x)
        hits.append(pair_hit)
        n_pairs = pair_hit.shape[0]
    if n_planes:
        plane_gap = x @ cp.plane_normal.T - cp.plane_offset - cp.radius[:, None]  # (ns, np)
        hits.append((plane_gap <= 0.0).reshape(-1))
    hit = jnp.concatenate(hits)

    capacity = cfg.capacity
    count = jnp.sum(hit)
    short = capacity - hit.shape[0]
    if short > 0:  # fewer candidates than buffer slots: pad so that selection has capacity items
        hit = jnp.concatenate((hit, jnp.zeros(short, bool)))

    def first():
        # compaction by prefix sum and scatter (faster than ``jnp.nonzero(size=...)``); hits beyond
        # the capacity go to a dropped sink slot
        rank = jnp.cumsum(hit, dtype=jnp.int32) - 1
        slot = jnp.where(hit & (rank < capacity), rank, capacity)
        every = jnp.arange(hit.shape[0], dtype=jnp.int32)
        return jnp.zeros(capacity + 1, jnp.int32).at[slot].set(every)[:capacity]

    def deepest():
        # slow path, only taken on overflow: rank every hit by its true gap
        gaps = []
        if ns >= 2:
            xs, radius = x[order], cp.radius[order]
            dist = jnp.linalg.norm(xs[pos_b] - xs[pos_a], axis=-1)
            gaps.append(dist - radius[pos_a] - radius[pos_b])
        if n_planes:
            gaps.append(plane_gap.reshape(-1))
        gap = jnp.concatenate(gaps + [jnp.zeros(max(short, 0), x.dtype)])
        return lax.top_k(jnp.where(hit, -gap, -jnp.inf), capacity)[1].astype(jnp.int32)

    idx = lax.cond(count > capacity, deepest, first)
    kept = jnp.minimum(count, capacity)
    live = jnp.arange(capacity) < kept

    # geometry of the selected contacts: sphere pair candidates come first, plane candidates after
    is_plane = idx >= n_pairs
    plane_flat = jnp.where(is_plane, idx - n_pairs, 0)
    plane_sphere, plane_id = plane_flat // max(n_planes, 1), plane_flat % max(n_planes, 1)
    sphere_a = sphere_b = plane_sphere
    if ns >= 2:
        pair = jnp.where(is_plane, 0, idx)
        sphere_a, sphere_b = order[pos_a[pair]], order[pos_b[pair]]
    sphere_a, sphere_b = (jnp.where(is_plane, plane_sphere, s) for s in (sphere_a, sphere_b))

    x_b, r_b = x[sphere_b], cp.radius[sphere_b]
    sdf, grad = ct.sphere_sdf(jnp, x_b, x[sphere_a], cp.radius[sphere_a])
    rest = jnp.minimum(cp.restitution[sphere_a], cp.restitution[sphere_b])
    if n_planes:
        sdf_p, grad_p = ct.plane_sdf(jnp, x_b, cp.plane_normal[plane_id], cp.plane_offset[plane_id])
        sdf = jnp.where(is_plane, sdf_p, sdf)
        grad = jnp.where(is_plane[:, None], grad_p, grad)
        rest = jnp.where(
            is_plane, jnp.minimum(cp.restitution[sphere_b], cp.plane_restitution[plane_id]), rest
        )
    gap, normal, point = ct.sphere_vs_sdf(jnp, x_b, r_b, sdf, grad)
    body_a = jnp.where(is_plane, -1, cp.sphere[sphere_a])
    return ContactBuffer(
        jnp.where(live, body_a, -1),
        jnp.where(live, cp.sphere[sphere_b], 0),
        point,
        normal,
        gap,
        rest,
        kept,
        (count > capacity) | cell_overflow,
    )


def solve(cp: ContactParams, cfg: ContactSettings, dt, contacts: ContactBuffer, u0, u, correct):
    """Velocities ``u`` (n, 6) after the contact impulses; ``u0`` is the start-of-step velocity.

    ``u`` must already satisfy the joints. ``correct(x) -> (x, change)`` re-imposes the joint
    constraints on the extended velocities (n + 1, 6), or None for scenes without joints.
    """
    n = u.shape[0]
    capacity = cfg.capacity
    live = jnp.arange(capacity) < contacts.count
    a = jnp.where(contacts.body_a < 0, n, contacts.body_a)
    b = contacts.body_b
    normal = contacts.normal

    v0 = jnp.concatenate((u0[:, :3], jnp.zeros((1, 3), u0.dtype)))
    approach = jnp.sum(normal * (v0[b] - v0[a]), axis=-1)
    rhs = jnp.where(approach < -cfg.restitution_threshold, -contacts.restitution * approach, 0.0)
    if cfg.stabilization > 0.0:
        rhs = jnp.maximum(rhs, -cfg.stabilization * contacts.gap / dt)

    inv_a, inv_b = cp.inv_mass[a], cp.inv_mass[b]
    weight = live.astype(u.dtype)
    per_body = jnp.zeros(n + 1, u.dtype).at[a].add(weight).at[b].add(weight).at[n].set(0.0)
    share = jnp.where(live, jnp.maximum(per_body[a], per_body[b]) * (inv_a + inv_b), 1.0)
    scale = jnp.where(live, cfg.omega / share, 0.0)
    push_a, push_b = inv_a[:, None] * normal, inv_b[:, None] * normal

    dim = 3 if correct is None else 6
    lin = slice(None) if dim == 3 else slice(0, 3)  # linear columns of the iterated velocities
    x0 = jnp.concatenate((u[:, :dim], jnp.zeros((1, dim), u.dtype)))

    def sweep(state):
        it, lam, x, _ = state
        rel = jnp.sum(normal * (x[b, lin] - x[a, lin]), axis=-1)
        new = jnp.maximum(0.0, lam - scale * (rel - rhs))
        delta = (new - lam)[:, None]
        x = x.at[a, lin].add(-delta * push_a).at[b, lin].add(delta * push_b)
        change = jnp.max(jnp.abs(delta))
        if correct is not None:
            x, joint_change = correct(x)
            change = jnp.maximum(change, joint_change)
        return it + 1, new, x, change

    def keep_going(state):
        it, _, _, change = state
        return (it < cfg.iterations) & (change >= cfg.tolerance)

    inf = jnp.asarray(jnp.inf, u.dtype)
    init = (jnp.zeros((), jnp.int32), jnp.zeros(capacity, u.dtype), x0, inf)
    x = lax.while_loop(keep_going, sweep, init)[2][:n]
    return x if correct is not None else u.at[:, :3].set(x)
