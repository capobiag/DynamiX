"""Block-sparse joint system with a block-banded Delassus matrix (JAX, functional).

Same formulation as ``dynamix.numpy_backend.joint_system``: each joint has one *side* per
attached body (the world has none) and ``rows`` equations (see ``dynamix.core.joints``);

    J_side = [ s E,  ang ]        (rows x 6; E is replaced by a ``lin`` block for prismatic joints)

and G = J M^-1 J^T is assembled from per-side blocks. Differences from the NumPy version, made
for speed under ``jit``: the topology is an immutable PyTree of index arrays built once, the
band is stored as ``rows`` x ``rows`` blocks ``band[d, j] = G[j + d, j]`` (see ``banded.py``),
and scatters are ``.at[].add`` operations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dynamix.core import joint_math as jm
from dynamix.core import joints as jt
from dynamix.core.joint_topology import JointTopology, build_joint_topology

BallJointSystem = JointSystem = JointTopology  # the same fields, holding jax arrays


def make_joint_system(*args, **kwargs):
    """Build the topology PyTree on the host; returns ``(system, block_half_bandwidth)``."""
    topology = build_joint_topology(*args, **kwargs)
    return jax.tree_util.tree_map(jnp.asarray, topology), topology.block_bandwidth


def update(system, q, rot):
    """Return ``(g, jw, lin)``: residual (nj, rows), angular blocks (n_sides, rows, 3), linear
    blocks (nj, rows, 3) of the b side or None when they are E (see ``joints.evaluate``)."""
    g, ang, lin = jt.evaluate(jnp, system.joints, q, rot)
    return g, ang[system.side_ang], lin


def delassus_band(system, jw, lin, block_bandwidth):
    """G = J M^-1 J^T in lower block-banded storage ``(m + 1, nj, rows, rows)``; also ``ww``."""
    n_joints, rows = system.joints.n_joints, system.joints.rows
    ww = system.inertia_inv_side @ jw.transpose(0, 2, 1)
    blocks = jw[system.pair_s] @ ww[system.pair_t]
    if lin is None:
        blocks = blocks + system.pair_coeff[:, None, None] * jnp.eye(rows, dtype=blocks.dtype).at[
            jt.POSITION_ROWS :, jt.POSITION_ROWS :
        ].set(0.0)
    else:
        lin_s = lin[system.side_joint[system.pair_s]]
        lin_t = lin[system.side_joint[system.pair_t]]
        blocks = blocks + system.pair_coeff[:, None, None] * (lin_s @ lin_t.transpose(0, 2, 1))
    band = jnp.zeros((block_bandwidth + 1, n_joints, rows, rows), blocks.dtype)
    band = band.at[system.pair_band, system.pair_col].add(blocks)
    if system.pad_diag is not None:
        band = band.at[0].add(system.pad_diag[:, :, None] * jnp.eye(rows, dtype=blocks.dtype))
    return band, ww


def jacobian_times(system, jw, lin, u):
    """J u as an array of shape (nj, rows)."""
    body = system.side_body
    lin_side = None if lin is None else lin[system.side_joint]
    rate = jm.side_rate(jnp, jw, system.sign, u[body, :3], u[body, 3:], lin_side)
    return (
        jnp.zeros((system.joints.n_joints, jw.shape[1]), rate.dtype).at[system.side_joint].add(rate)
    )


def add_impulse(system, lin, ww, impulse, u):
    """Return ``u + M^-1 J^T impulse``; ``impulse`` has shape (n_joints, rows)."""
    local = impulse[system.side_joint]
    if lin is None:
        d_lin = system.inv_mass_signed[:, None] * local[:, : jt.POSITION_ROWS]
    else:
        lin_t = lin[system.side_joint].transpose(0, 2, 1)
        d_lin = system.inv_mass_signed[:, None] * (lin_t @ local[:, :, None])[:, :, 0]
    d_ang = (ww @ local[:, :, None])[:, :, 0]
    return u.at[system.side_body].add(jnp.concatenate((d_lin, d_ang), axis=1))
