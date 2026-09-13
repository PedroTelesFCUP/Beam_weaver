"""Beam Weaver: coordinates."""

# Copyright (c) 2025–2026 Pedro Teles & João Melo. Apache-2.0.

from .constants import (
    KEV_PER_MEV,
)

from .constants import BW4_EMAX_MEV
from .constants import BW4_EMIN_MEV
from .constants import HC_KEV_A
from .constants import mec2
import numpy as np


def norm_logE(E):
    E = np.asarray(E, np.float64)
    return 2.0 * (np.log10(E / BW4_EMIN_MEV)
                  / np.log10(BW4_EMAX_MEV / BW4_EMIN_MEV)) - 1.0


def k_invA(E):
    return np.asarray(E, np.float64) * KEV_PER_MEV / HC_KEV_A


def q2max(E):
    return 4.0 * k_invA(E) ** 2


def q2_from_cos(c, E):
    return 2.0 * k_invA(E) ** 2 * (1.0 - np.asarray(c, np.float64))


def cos_from_q2(q2, E):
    return 1.0 - np.asarray(q2, np.float64) / (2.0 * k_invA(E) ** 2)


def lq2r_from_cos(c):
    """Support-normalized Rayleigh coordinate s = ln((1-cos t)/2)
    = ln(q2/q2_max): identical support (-inf, 0] at EVERY energy, so global
    quantile edges resolve all energies and no live-bin mask exists."""
    return np.log(np.maximum((1.0 - np.asarray(c, np.float64)) / 2.0,
                             1e-30))


def cos_from_lq2r(sv):
    return 1.0 - 2.0 * np.exp(np.asarray(sv, np.float64))


def tau_min(E):
    return 1.0 / (1.0 + 2.0 * np.asarray(E, np.float64) / mec2)


def u_from_tau(tau, E):
    return (1.0 - np.asarray(tau, np.float64)) / (1.0 - tau_min(E))


def tau_from_u(u, E):
    return 1.0 - np.asarray(u, np.float64) * (1.0 - tau_min(E))


def cos_from_tau(tau, E):
    tau = np.asarray(tau, np.float64)
    return 1.0 - (1.0 - tau) / ((np.asarray(E, np.float64) / mec2) * tau)


def transform_selftest(n=20000, seed=40001, tol=1e-10):
    rng = np.random.default_rng(seed)
    E = np.exp(rng.uniform(np.log(BW4_EMIN_MEV), np.log(BW4_EMAX_MEV), n))
    c = rng.uniform(-1, 1, n)
    worst = max(
        float(np.max(np.abs(cos_from_q2(q2_from_cos(c, E), E) - c))),
        float(np.max(np.abs(cos_from_lq2r(lq2r_from_cos(c)) - c))),
        float(abs(lq2r_from_cos(-1.0) - 0.0)),
        float(np.max(np.abs(q2_from_cos(-1.0, E) - q2max(E)))),
        float(np.max(np.abs(tau_from_u(
            u_from_tau(tau_min(E) + (1 - tau_min(E)) * c**2, E),
            E) - (tau_min(E) + (1 - tau_min(E)) * c**2)))),
        float(np.max(np.abs(cos_from_tau(tau_min(E), E) + 1.0))),
        float(abs(norm_logE(BW4_EMIN_MEV) + 1.0)
              + abs(norm_logE(BW4_EMAX_MEV) - 1.0)))
    assert worst < tol, f"transform self-test failed: {worst}"
    return worst
