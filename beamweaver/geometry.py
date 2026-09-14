"""Homogeneous water phantom and the current policy energy encoding."""

# Copyright (c) 2025–2026 Pedro Teles. Apache-2.0.

import math
import random

import numpy as np

from .constants import (
    BW4_EMAX_MEV,
    BW4_EMIN_MEV,
    DEFAULT_TRANSPORT_ECUT_MEV,
    DIRECTION_FRAME_SWITCH,
    SOURCE_FIELD_HALF_WIDTH_CM,
    SOURCE_FIELD_WIDTH_CM,
)


class WaterPhantom:
    """Water box in cm, with a uniform 10 × 10 cm² incident photon field.

    The 100 depth slabs are 1 cm thick. ``dose_tally`` accumulates deposited
    energy in MeV; reporting can produce a normalized PDD. Absorbed dose
    requires a separate mass normalization.
    """

    def __init__(self, data, fixed_energy=0.1,
                 ecut=DEFAULT_TRANSPORT_ECUT_MEV,
                 energy_range=(BW4_EMIN_MEV, BW4_EMAX_MEV)):
        self.data = data
        self.fixed_energy = fixed_energy
        self.ecut = ecut
        self.E_min, self.E_max = energy_range
        self.xmin, self.xmax = -50.0, 50.0
        self.ymin, self.ymax = -50.0, 50.0
        self.zmin, self.zmax = 0.0, 100.0
        self.pdd_bins = 100
        self.dz = (self.zmax - self.zmin) / self.pdd_bins
        self.dose_tally = np.zeros(self.pdd_bins, dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None):
        """Reset the photon state and deposited-energy tally.

        Keep the two source-position draws here: the MC runner resets the
        phantom before drawing its primary histories, and the draw order
        is part of reproducible seeded transport.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.steps = 0
        self.x = (random.random() * SOURCE_FIELD_WIDTH_CM) - SOURCE_FIELD_HALF_WIDTH_CM
        self.y = (random.random() * SOURCE_FIELD_WIDTH_CM) - SOURCE_FIELD_HALF_WIDTH_CM
        self.z = 0.0
        self.initial_position = (self.x, self.y, self.z)
        self.u, self.v, self.w = (0.0, 0.0, 1.0)
        self.E = self.fixed_energy
        self.alive = True
        self.dose_tally[:] = 0.0

    def _extract_local_phi(self, inc_dir, new_dir, theta):
        """
        Recover the local azimuthal scattering angle φ from the
        incident and scattered direction vectors.
        This is the inverse of rotate_direction(inc_dir, theta, phi).
        """
        # rotate_direction (|uz| < DIRECTION_FRAME_SWITCH) builds the orthonormal local
        # frame
        #     â = (ux·uz/denom, uy·uz/denom, −denom),  b̂ = (−uy/denom, ux/denom, 0),
        #     denom = √(1 − uz²)
        # and returns  new = cosθ·û + sinθ·(cosφ·â + sinφ·b̂).  Projecting the
        # scattered direction onto (â, b̂) therefore recovers (sinθ·cosφ,
        # sinθ·sinφ) and φ = atan2(·, ·) EXACTLY — with no division by ux.
        st = math.sin(theta)
        if st < 1e-12:
            return 0.0  # forward/backward — φ undefined
        ux, uy, uz = inc_dir
        if abs(uz) < DIRECTION_FRAME_SWITCH:
            denom = math.sqrt(1.0 - uz * uz)
            a_hat = (ux * uz / denom, uy * uz / denom, -denom)
            b_hat = (-uy / denom, ux / denom, 0.0)
            cp = (new_dir[0] * a_hat[0] + new_dir[1] * a_hat[1]
                  + new_dir[2] * a_hat[2])
            sp = (new_dir[0] * b_hat[0] + new_dir[1] * b_hat[1]
                  + new_dir[2] * b_hat[2])
            phi = math.atan2(sp, cp)
        else:
            # Polar branch of rotate_direction: new = (sinθ·cosφ, sinθ·sinφ, ±cosθ)
            # for BOTH signs of uz, so φ = atan2(v₂, u₂) directly.
            phi = math.atan2(new_dir[1], new_dir[0])
        return phi % (2 * math.pi)


def encode_policy_energy(E, env):
    """Return MeV energy after the current float32 log-energy encoding.

    The deployed kernel previously received this value through an
    observation vector. Keep its scalar rounding and clipping exactly
    while eliminating the unused observation features.
    """
    E_val = max(E, BW4_EMIN_MEV)
    logE = np.clip(math.log10(max(E_val, env.E_min)),
                   math.log10(env.E_min), math.log10(env.E_max))
    return 10.0 ** float(np.float32(logE))


def encode_policy_energy_batch(energies, env):
    """Vectorized current log-energy encoding; input/output are MeV arrays.

    Preserve NumPy's log10 and the float32-to-float64 conversion used by
    batched inference, which can round differently from the scalar path.
    """
    E = np.asarray(energies, dtype=float)
    E_val = np.maximum(E, BW4_EMIN_MEV)
    logE = np.clip(np.log10(np.maximum(E_val, env.E_min)),
                   math.log10(env.E_min), math.log10(env.E_max))
    return 10.0 ** logE.astype(np.float32).astype(np.float64)
