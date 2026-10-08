"""Contact detection (NumPy): grid broad phase plus SDF narrow phase for spheres and planes.

``ContactDetector.detect(q)`` returns a ``ContactBuffer`` (see ``dynamix.core.contacts``) holding
the contacts with ``gap <= 0`` at the configuration ``q``. The buffer arrays are allocated once
and reused; the returned views are valid until the next call.
"""

from __future__ import annotations

import warnings

import numpy as np

from dynamix.core import contacts as ct
from dynamix.core.buffers import SystemBuffers

# Offsets of the 13 "forward" neighbour cells; together with the own cell they cover every pair of
# adjacent cells exactly once.
_FORWARD = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
]


def all_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Every pair ``i < j`` (O(n^2); reference for the grid)."""
    i, j = np.triu_indices(n, k=1)
    return i.astype(np.intp), j.astype(np.intp)


def grid_pairs(x: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Candidate pairs ``(i, j)`` of points in the same or adjacent cells of edge ``cell``.

    Every pair closer than ``cell`` is returned exactly once (as ``i != j``, order not fixed).
    """
    n = x.shape[0]
    if n < 2:
        return np.empty(0, np.intp), np.empty(0, np.intp)
    index = np.floor((x - x.min(axis=0)) / cell).astype(np.int64) + 1  # >= 1: room for -1 offsets
    dims = index.max(axis=0) + 2
    key = (index[:, 0] * dims[1] + index[:, 1]) * dims[2] + index[:, 2]
    order = np.argsort(key, kind="stable")
    cells, start, count = np.unique(key[order], return_index=True, return_counts=True)

    first, second = [], []
    for off in [(0, 0, 0), *_FORWARD]:
        target = key + (off[0] * dims[1] + off[1]) * dims[2] + off[2]
        pos = np.minimum(np.searchsorted(cells, target), cells.shape[0] - 1)
        hit = np.flatnonzero(cells[pos] == target)
        if hit.size == 0:
            continue
        cnt = count[pos[hit]]
        total = int(cnt.sum())
        within = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        i = np.repeat(hit, cnt)
        j = order[np.repeat(start[pos[hit]], cnt) + within]
        if off == (0, 0, 0):
            keep = j > i
            i, j = i[keep], j[keep]
        first.append(i)
        second.append(j)
    if not first:
        return np.empty(0, np.intp), np.empty(0, np.intp)
    return np.concatenate(first).astype(np.intp), np.concatenate(second).astype(np.intp)


class ContactDetector:
    """Sphere-sphere (grid) and sphere-plane (SDF) contacts into a fixed-capacity buffer."""

    def __init__(self, buffers: SystemBuffers, use_grid: bool = True):
        self.spheres = np.flatnonzero(buffers.collider_radius > 0.0)
        self.radius = buffers.collider_radius[self.spheres]
        self.restitution = buffers.collider_restitution[self.spheres]
        self.plane_normal = buffers.plane_normal
        self.plane_offset = buffers.plane_offset
        self.plane_restitution = buffers.plane_restitution
        self.capacity = buffers.max_contacts
        self.use_grid = use_grid
        self.cell = 2.0 * float(self.radius.max()) if self.radius.size else 1.0
        self.buffer = ct.empty_buffer(self.capacity)
        self._warned = False

    def candidate_pairs(self, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.use_grid:
            return grid_pairs(centres, self.cell)
        return all_pairs(centres.shape[0])

    def _sphere_pairs(self, centres):
        i, j = self.candidate_pairs(centres)
        sdf, grad = ct.sphere_sdf(np, centres[j], centres[i], self.radius[i])
        gap, normal, point = ct.sphere_vs_sdf(np, centres[j], self.radius[j], sdf, grad)
        hit = gap <= 0.0
        i, j = i[hit], j[hit]
        rest = np.minimum(self.restitution[i], self.restitution[j])
        return self.spheres[i], self.spheres[j], point[hit], normal[hit], gap[hit], rest

    def _sphere_planes(self, centres):
        if self.plane_offset.shape[0] == 0:
            return None
        # (spheres, planes) SDF grid; hits are listed sphere-major
        x = centres[:, None, :]
        sdf, grad = ct.plane_sdf(np, x, self.plane_normal[None], self.plane_offset[None])
        gap, normal, point = ct.sphere_vs_sdf(np, x, self.radius[:, None], sdf, grad)
        s, p = np.nonzero(gap <= 0.0)
        rest = np.minimum(self.restitution[s], self.plane_restitution[p])
        return (
            np.full(s.shape[0], -1, dtype=np.intp),
            self.spheres[s],
            point[s, p],
            normal[s, p],
            gap[s, p],
            rest,
        )

    def detect(self, q: np.ndarray) -> ct.ContactBuffer:
        centres = q[self.spheres, :3]
        parts = [self._sphere_pairs(centres)]
        planes = self._sphere_planes(centres)
        if planes is not None:
            parts.append(planes)
        a, b, point, normal, gap, rest = (np.concatenate(x) for x in zip(*parts, strict=True))

        overflow = a.shape[0] > self.capacity
        if overflow:
            keep = np.sort(np.argsort(gap, kind="stable")[: self.capacity])
            a, b, point, normal, gap, rest = (x[keep] for x in (a, b, point, normal, gap, rest))
            if not self._warned:
                warnings.warn("contact buffer overflow: keeping the deepest contacts", stacklevel=2)
                self._warned = True
        count = a.shape[0]
        buf = self.buffer
        buf.body_a[:count], buf.body_b[:count] = a, b
        buf.point[:count], buf.normal[:count] = point, normal
        buf.gap[:count], buf.restitution[:count] = gap, rest
        self.buffer = buf._replace(count=count, overflow=overflow)
        return self.buffer
