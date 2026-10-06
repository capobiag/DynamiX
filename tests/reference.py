"""Independent Lagrangian reference for a planar chain of uniform rods (n-fold pendulum)."""

import numpy as np
from scipy.integrate import solve_ivp


def _mass_and_potential_terms(lengths, masses, g):
    n = len(lengths)
    # coeff[k, j] = d(COM_k)/d(phi_j) magnitude
    coeff = np.zeros((n, n))
    for k in range(n):
        coeff[k, :k] = lengths[:k]
        coeff[k, k] = 0.5 * lengths[k]
    a = (masses[:, None, None] * coeff[:, :, None] * coeff[:, None, :]).sum(axis=0)
    inertia = masses * lengths**2 / 12.0
    grad_v = g * (masses[:, None] * coeff).sum(axis=0)  # dV/dphi_i = grad_v[i] * sin(phi_i)
    return a, inertia, grad_v


def simulate_chain(phi0, lengths, masses, t_eval, g=9.81):
    """Integrate absolute angles phi (about +y from the downward vertical) from rest."""
    lengths, masses = np.asarray(lengths, float), np.asarray(masses, float)
    a, inertia, grad_v = _mass_and_potential_terms(lengths, masses, g)
    n = len(lengths)

    def rhs(_, y):
        phi, omega = y[:n], y[n:]
        diff = phi[:, None] - phi[None, :]
        mass_matrix = a * np.cos(diff) + np.diag(inertia)
        force = -(a * np.sin(diff)) @ omega**2 - grad_v * np.sin(phi)
        return np.concatenate((omega, np.linalg.solve(mass_matrix, force)))

    y0 = np.concatenate((np.broadcast_to(phi0, (n,)), np.zeros(n)))
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, rtol=1e-11, atol=1e-12)
    return sol.y[:n].T
