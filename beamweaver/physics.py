"""Beam Weaver: physics."""

# Copyright (c) 2025–2026 Pedro Teles & João Melo. Apache-2.0.

import os
from .constants import PAIR_THRESHOLD_MEV
from .constants import (
    ATTENUATION_FLOOR_CM_INV,
    DEFAULT_TRANSPORT_ECUT_MEV,
    DIRECTION_FRAME_SWITCH,
    ELECTRON_STOPPING_POWER_TABLE,
    HC_MEV_A,
    INCOHERENT_SCATTERING_TABLE,
    KEV_PER_MEV,
    N_SHELLS,
    PHOTO_SHELL_BINDINGS_MEV,
    SHELL_NAMES,
)
from .materials import WaterPhotonData
from .constants import _PAIR_ALPHA_FS
from .constants import _PAIR_R_EQ
from .constants import _PAIR_Z_EQ
_SAUTER_ENV_CACHE = {}  # Computed values cached for this module.
import math
from .constants import mec2
import numpy as np
import pandas as pd
import random


class ComptonEnergyTransferSampler:
    """Sample Compton energy transfer using the incoherent scattering table.

    ``sample_energy_transfer()`` returns (electron kinetic energy in MeV,
    outgoing photon energy / incident photon energy). Direction and recoil
    reconstruction belong to ``sample_compton_event()``.
    """
    _TABLE_CACHE = {}   # Parse water_sq.csv once per path;
                        # dataset generation / probes construct thousands of
                        # samplers; this removes repeated file parsing only.

    def __init__(self, E_mev, sq_csv=INCOHERENT_SCATTERING_TABLE):
        self.E = E_mev
        self.alpha = E_mev / mec2
        self.tau = 2.0 * self.alpha
        self.eps0 = 1.0 / (1.0 + self.tau)
        self.eps0_2 = self.eps0 ** 2
        self.a1 = math.log(1.0 + self.tau)
        # Weights normalise the two branches of g(eps) ∝ 1/eps + eps.
        self.a2 = 0.5 * (1.0 - self.eps0_2)
        # Load S(q) table (q = sin(theta/2)/lambda * 2) — cached per path
        key = os.path.abspath(sq_csv)
        if key not in ComptonEnergyTransferSampler._TABLE_CACHE:
            data = np.genfromtxt(sq_csv, delimiter=',', names=True)
            order = np.argsort(data['q'])
            ComptonEnergyTransferSampler._TABLE_CACHE[key] = (
                np.asarray(data['q'][order], dtype=float),
                np.asarray(data['S_q'][order], dtype=float))
        self.q_grid, self.S_grid = ComptonEnergyTransferSampler._TABLE_CACHE[key]
        self.S_max = 10.0  # For water (Z=10)

    def compute_q(self, eps, cost):
        """Calculate momentum transfer q (Å⁻¹) using relativistic formula."""
        E_out = self.E * eps
        return np.sqrt(
            self.E**2 + E_out**2 - 2 * self.E * E_out * cost
        ) / HC_MEV_A  # Correct unit conversion

    def sample_energy_transfer(self):
        """Sample energy transfer with PENELOPE's rejection function."""
        for _ in range(1000):  # Max attempts
            # 1. Generate eps using PENELOPE's composition method
            if random.random() < self.a1/(self.a1 + self.a2):
                eps = math.exp(random.random() * self.a1) * self.eps0
            else:
                eps = math.sqrt(self.eps0_2 + random.random()*(1 - self.eps0_2))
            
            # 2. Compute scattering angle variables
            cost = 1.0 - (1.0 - eps) / (eps * self.alpha)
            cost = np.clip(cost, -1.0, 1.0)
            theta = math.acos(cost)
            
            # 3. Compute momentum transfer and table x
            q = self.compute_q(eps, cost)
            x_table = q / 2  # Convert to table's x = sin(theta/2)/lambda
            
            # 4. Interpolate S(q)
            S_q = np.interp(
                x_table, 
                self.q_grid, 
                self.S_grid, 
                left=0.0, 
                right=self.S_max
            )
            
            # Klein–Nishina acceptance for the 1/eps + eps proposal:
            # T = 1 - eps*sin(theta)^2/(1 + eps^2). Multiplication by S(q)
            # includes the tabulated incoherent scattering function.
            sin2 = max(0.0, 1.0 - cost * cost)
            T_kn = 1.0 - eps * sin2 / (1.0 + eps * eps)

            # 6. Combined acceptance probability
            accept_prob_compt = (S_q / self.S_max) * T_kn
            if random.random() <= accept_prob_compt:
                return (self.E*(1 - eps), eps)  # (T_meV, eps)
        
        # Fallback after max attempts
        return (self.E*(1 - self.eps0), self.eps0)



def rotate_direction(old_dir, theta, phi):
    """
    Rotate a direction vector by theta and phi angles.
    
    Args:
        old_dir: Original direction vector (3-element iterable)
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
        
    Returns:
        New direction vector (numpy array) after rotation
    """
    (u, v, w) = old_dir
    mag = math.sqrt(u*u + v*v + w*w)
    if mag < 1e-14:
        return np.array([
            math.sin(theta)*math.cos(phi),
            math.sin(theta)*math.sin(phi),
            math.cos(theta)
        ])
    
    ux = u / mag
    uy = v / mag
    uz = w / mag
    st = math.sin(theta)
    ct = math.cos(theta)
    sp = math.sin(phi)
    cp = math.cos(phi)
    
    if abs(uz) < DIRECTION_FRAME_SWITCH:
        denom = math.sqrt(1.0 - uz*uz)
        u2 = ux*ct + ((ux*uz*cp - uy*sp) / denom) * st
        v2 = uy*ct + ((uy*uz*cp + ux*sp) / denom) * st
        w2 = uz*ct - denom*cp*st
    else:
        u2 = st*cp
        v2 = st*sp
        w2 = ct if uz > 0 else -ct
    
    return np.array([u2, v2, w2])


def sample_rayleigh_event(E, old_dir, data):
    """
    Rayleigh (coherent) scattering using the tabulated molecular form
    factor:      dσ/dΩ ∝ (1 + cos²θ) · |F(q)|² ,   q = 2k sin(θ/2).

    Method: inverse-CDF sampling of F²(q) in the q² variable + Thomson
    rejection.
      1) Draw q² from the density ∝ F²(q) dq², restricted to the kinematic
         range q ≤ 2k, by inverting the precomputed cumulative
         ∫F²(q) dq² (data._ray_cdf, built at init from the same
         piecewise-linear F(x) that coherent_form_factor() interpolates).
      2) cosθ = 1 − q²/(2k²);  accept with the Thomson polarisation factor
         (1 + cos²θ)/2.
    The inverse CDF uses numerical quadrature and interpolation of the
    tabulated form factor. A non-positive accessible CDF or exhaustion of
    64 rejection attempts returns a forward event. Those exceptional paths
    are retained and should be assessed with the production tables.
    """
    shell_onehot = [0] * N_SHELLS  # No shell activation for Rayleigh

    # Wave number in Å⁻¹;  Hubbell table variable x = q/2 ≤ k
    EkeV = E * KEV_PER_MEV          # MeV → keV
    k = EkeV / data.HC_KEV_A

    xg, cdf = data._ray_x_grid, data._ray_cdf
    # CDF value at the kinematic limit x_max = k  (q_max = 2k)
    A = float(np.interp(min(k, xg[-1]), xg, cdf))
    if A <= 0.0:                    # k below the first table knot — forward
        phi = 2.0 * math.pi * random.random()
        return (rotate_direction(old_dir, 0.0, phi), E, [], "rayleigh",
                shell_onehot)

    for _ in range(64):
        # ── Step 1: inverse-transform x (⇔ q²) from ∝ F²(q) dq² ──
        u = random.random() * A
        x = float(np.interp(u, cdf, xg))
        # ── Step 2: kinematics — sin²(θ/2) = (x/k)² ──
        s2 = min((x / k) ** 2, 1.0)
        cost = 1.0 - 2.0 * s2
        # ── Step 3: Thomson polarisation rejection, efficiency ≥ 1/2 ──
        if random.random() <= 0.5 * (1.0 + cost * cost):
            theta = math.acos(max(-1.0, min(1.0, cost)))
            phi = 2.0 * math.pi * random.random()
            new_dir = rotate_direction(old_dir, theta, phi)
            return (new_dir, E, [], "rayleigh", shell_onehot)

    # Exhausted rejection loop: retain the implemented forward fallback.
    phi = 2.0 * math.pi * random.random()
    new_dir = rotate_direction(old_dir, 0.0, phi)
    return (new_dir, E, [], "rayleigh", shell_onehot)


def sample_compton_event(E, old_dir, data=None, sampler=None):
    """
    Sample a Compton scattering event using PENELOPE methodology.
    
    Args:
        E: Photon energy in MeV
        old_dir: Incident photon direction (3D vector)
        data: Additional data for physics calculations (optional)
        
    Returns:
        Tuple of (new_dir, Eout, secondaries, process_name, shell_info)
    """
    # Reuse a sampler when generating repeated events at the same energy.
    comp = sampler if sampler is not None else ComptonEnergyTransferSampler(E)
    T, eps = comp.sample_energy_transfer()
    
    # Recover the polar angle from the sampled energy ratio.
    alpha = E / mec2
    cost = 1.0 - (1.0 - eps) / (eps * alpha)
    
    # Clamp cosine to valid range (-1 to 1)
    cost_clamped = max(-1.0, min(1.0, cost))
    
    # Sample azimuthal angle (uniform in 0 to 2π)
    phi = 2.0 * math.pi * random.random()
    
    # Calculate polar angle
    theta = math.acos(cost_clamped)
    
    # Sanity check for numerical issues
    if not np.isfinite(theta):
        print(f"Warning: Non-finite theta. cost={cost}, eps={eps}, alpha={alpha}")
        theta = 0.0  # Safe default
    
    # Compute new photon direction
    new_dir = rotate_direction(old_dir, theta, phi)
    
    # Outgoing photon energy
    Eout = E * eps  # or equivalently: E - T
    
    # Calculate electron recoil direction using momentum conservation
    electron_mom = np.array(old_dir) * E - new_dir * Eout
    norm = np.linalg.norm(electron_mom)
    
    if norm < 1e-14:
        edir = np.array([0, 0, 0])
    else:
        edir = electron_mom / norm
    
    # Create secondaries list with electron information
    secs = [("electron", T, edir, "compton_e")]
    
    # Default shell onehot (no photoelectric event)
    shell_onehot = [0] * N_SHELLS
    
    return (new_dir, Eout, secs, "compton", shell_onehot)


def _sauter_dcs_shape(cos_theta, beta, gamma):
    """Unnormalised Sauter dσ/dΩ as a function of cosθ (scalar or ndarray)."""
    d    = 1.0 - beta * cos_theta
    sin2 = 1.0 - cos_theta * cos_theta
    val  = sin2 / d**4 * (1.0 + 0.5 * gamma * (gamma - 1.0) * (gamma - 2.0) * d)
    return np.maximum(val, 0.0) if isinstance(val, np.ndarray) else max(val, 0.0)


def _sauter_envelope_max(beta, gamma):
    """Grid-argmax envelope for uniform-in-cosθ rejection (cached per (β, γ))."""
    # Key on both parameters: beta and gamma are coupled
    # through T, but relying on that coupling to keep a β-only key safe
    # is a needless gamble.
    key = (round(beta, 6), round(gamma, 6))
    if key not in _SAUTER_ENV_CACHE:
        grid = np.linspace(-1.0, 1.0, 2001)
        _SAUTER_ENV_CACHE[key] = 1.05 * float(_sauter_dcs_shape(grid, beta, gamma).max())
    return _SAUTER_ENV_CACHE[key]


def sample_photoelectric_event_for_shell(E, old_dir, data, idx):
    """Shell-conditioned photoelectric sampling for the
    stratified dataset — identical angular physics to sample_photoelectric_event
    (Sauter for H_K/O_K, isotropic for L shells), shell imposed rather than
    drawn, so every shell-conditioned head cell receives real support."""
    shellName = SHELL_NAMES[idx]
    shell_onehot = [0] * N_SHELLS; shell_onehot[idx] = 1
    E_e = E - PHOTO_SHELL_BINDINGS_MEV[shellName]
    if E_e <= 0:
        return (old_dir, E, [], "photo_failed", shell_onehot)
    if shellName in ("H_K", "O_K"):
        gamma = 1.0 + E_e / mec2
        beta = math.sqrt(E_e * (E_e + 2.0 * mec2)) / (E_e + mec2)
        env_max = _sauter_envelope_max(beta, gamma)
        while True:
            cos_theta = 2.0 * random.random() - 1.0
            if random.random() * env_max <= _sauter_dcs_shape(cos_theta, beta, gamma):
                break
    else:
        cos_theta = 2.0 * random.random() - 1.0
    theta = math.acos(cos_theta); phi = 2.0 * math.pi * random.random()
    e_dir = rotate_direction(old_dir, theta, phi)
    secs = [("electron", E_e, e_dir, f"photo_{shellName}")]
    return (np.array([0.0, 0.0, 0.0]), 0.0, secs, "photo", shell_onehot)


def sample_photoelectric_event(E, old_dir, data):
    idx = data.sample_photo_shell_index(E)
    if idx is None:
        shell_onehot = [0] * N_SHELLS
        return (old_dir, E, [], "photo_none", shell_onehot)
    shellName = SHELL_NAMES[idx]
    shell_onehot = [0] * N_SHELLS
    shell_onehot[idx] = 1
    Eb_MeV = PHOTO_SHELL_BINDINGS_MEV[shellName]
    E_e = E - Eb_MeV
    if E_e <= 0:
        return (old_dir, E, [], "photo_failed", shell_onehot)
    
    secs = []
    # Determine angular distribution based on shell
    if shellName in ("H_K", "O_K"):
        # Sauter K-shell sampling. Uniform proposal in
        # cosθ, accept ∝ _sauter_dcs_shape → p(cosθ) ∝ dσ/dΩ, hence
        # p(θ) ∝ dσ/dΩ·sinθ — the polar-angle density used for reference validation
        # The rejection loop uses the shared Sauter angular density.
        gamma = 1.0 + E_e / mec2
        beta  = math.sqrt(E_e * (E_e + 2.0 * mec2)) / (E_e + mec2)
        env_max = _sauter_envelope_max(beta, gamma)
        while True:
            cos_theta = 2.0 * random.random() - 1.0
            if random.random() * env_max <= _sauter_dcs_shape(cos_theta, beta, gamma):
                break
        theta = math.acos(cos_theta)
    else:
        # Isotropic distribution for L-shells
        cos_theta = 2 * random.random() - 1
        theta = math.acos(cos_theta)
    
    phi = 2 * math.pi * random.random()
    e_dir = rotate_direction(old_dir, theta, phi)
    secs.append(("electron", E_e, e_dir, f"photo_{shellName}"))
    
    return (np.array([0.0, 0.0, 0.0]), 0.0, secs, "photo", shell_onehot)


def _pair_fC(Z):
    """High-energy Coulomb correction f_C(Z) of Davies et al. (1954) —
    PENELOPE-2018 Eq. (2.80), series in a² with a = αZ."""
    a2 = (_PAIR_ALPHA_FS * Z) ** 2
    return a2 * (1.0 / (1.0 + a2) + 0.202059 - 0.03693 * a2
                 + 0.00835 * a2**2 - 0.00201 * a2**3 + 0.00049 * a2**4
                 - 0.00012 * a2**5 + 0.00003 * a2**6)


def _pair_F0(kappa, Z):
    """Empirical low-energy correction F0(κ, Z) — PENELOPE-2018 Eq. (2.88)."""
    a = _PAIR_ALPHA_FS * Z
    s = math.sqrt(2.0 / kappa)            # (2/κ)^(1/2)
    return ((-1.774 - 12.10 * a + 11.18 * a * a) * s
            + (8.523 + 73.26 * a - 44.41 * a * a) * s * s
            - (13.52 + 121.1 * a - 96.41 * a * a) * s ** 3
            + (8.946 + 62.05 * a - 63.41 * a * a) * s ** 4)


def _pair_g1g2(b):
    """Screening functions g1(b), g2(b) — PENELOPE-2018 Eq. (2.87)."""
    b2 = b * b
    atanb = math.atan(1.0 / b)
    l1 = math.log(1.0 + b2)
    core = 4.0 - 4.0 * b * atanb - 3.0 * math.log(1.0 + 1.0 / b2)
    g1 = 7.0 / 3.0 - 2.0 * l1 - 6.0 * b * atanb - b2 * core
    g2 = 11.0 / 6.0 - 2.0 * l1 - 3.0 * b * atanb + 0.5 * b2 * core
    return g1, g2


def _pair_phi12(eps, kappa, R, g0):
    """φ1(ε), φ2(ε) = g_i(b) + g0(κ), clipped at zero (PENELOPE-2018: 'these
    functions are set equal to zero when they take negative values')."""
    b = R * (1.0 / (2.0 * kappa)) / (eps * (1.0 - eps))   # Eq. (2.79)
    g1, g2 = _pair_g1g2(b)
    return max(g1 + g0, 0.0), max(g2 + g0, 0.0)


def _pair_sample_eps(E):
    """
    Sample the reduced electron energy  ε = (E_- + me·c²)/E  from the
    Bethe-Heitler PDF  p(ε) = 2(1/2 − ε)²·φ1(ε) + φ2(ε),  ε ∈ (κ⁻¹, 1−κ⁻¹),
    using PENELOPE-2018's composition + rejection algorithm (Eqs. 2.91-2.96):

      u1 = (2/3)(1/2 − κ⁻¹)²·φ1(1/2),   u2 = φ2(1/2)
      (i)   pick i∈{1,2} with probabilities u_i/(u1+u2)
      (ii)  i=1: ε = 1/2 + (1/2 − κ⁻¹)(2ξ−1)^(1/3)   [signed cube root —
            the manual explicitly warns 2ξ−1 may be negative]
            i=2: ε = κ⁻¹ + (1/2 − κ⁻¹)·2ξ
      (iii-iv) accept with probability U_i(ε) = φ_i(ε)/φ_i(1/2), else repeat
    Manual-quoted efficiency: > 70% near threshold, ~95% at 1 GeV.
    """
    kappa = E / mec2
    inv_k = 1.0 / kappa
    half_w = 0.5 - inv_k                                   # (1/2 − κ⁻¹)
    g0 = (4.0 * math.log(_PAIR_R_EQ) - 4.0 * _pair_fC(_PAIR_Z_EQ)
          + _pair_F0(kappa, _PAIR_Z_EQ))                   # Eq. (2.87)
    phi1_h, phi2_h = _pair_phi12(0.5, kappa, _PAIR_R_EQ, g0)
    u1 = (2.0 / 3.0) * half_w * half_w * phi1_h            # Eq. (2.92)
    u2 = phi2_h
    if u1 + u2 <= 0.0:
        # Eq. (2.85) shifts the effective threshold slightly above 2me·c²;
        # just above 2me·c² both φ_i(1/2) can clip to zero.  Symmetric split
        # (ε = 1/2) — the produced leptons then share E_avail equally.
        return 0.5
    p1 = u1 / (u1 + u2)
    for _ in range(1000):        # bounded loop; manual efficiency ≥ 70%
        if random.random() < p1:                           # i = 1
            t = 2.0 * random.random() - 1.0
            eps = 0.5 + half_w * math.copysign(abs(t) ** (1.0 / 3.0), t)
            phi_ref = phi1_h
            idx = 1
        else:                                              # i = 2
            eps = inv_k + half_w * 2.0 * random.random()
            phi_ref = phi2_h
            idx = 2
        phi1_e, phi2_e = _pair_phi12(eps, kappa, _PAIR_R_EQ, g0)
        U = (phi1_e if idx == 1 else phi2_e) / phi_ref     # Eq. (2.94)
        if random.random() <= U:
            return eps
    return 0.5                   # fallback after rejection-loop exhaustion


def _pair_lepton_costheta(E_kin):
    """
    Lepton polar angle — PENELOPE-2018 Eqs. (2.97)-(2.99):
        p(cosθ±) = a(1 − β±·cosθ±)⁻²,   β± = sqrt(E±(E± + 2me·c²))/(E± + me·c²)
        cosθ± = (2ξ − 1 + β±)/((2ξ − 1)β± + 1)   (exact inverse transform,
    bounded in [−1,1] by construction).
    """
    Etot = E_kin + mec2
    beta = math.sqrt(max(E_kin * (E_kin + 2.0 * mec2), 0.0)) / Etot
    xi = 2.0 * random.random() - 1.0
    return (xi + beta) / (xi * beta + 1.0)


def sample_pair_event(E, old_dir, data):
    """
    γ → e⁻e⁺ event, PENELOPE-2018 §2.4.1.  All events are simulated as
    nuclear pairs even though the tabulated channel (`pair_triplet` in
    Final_cross_sections.csv) includes triplet production — the manual's own
    treatment ('all the events are simulated as if they were pairs',
    justified because the triplet recoil electron's range is much smaller
    than the photon mean free path).  The triplet inner-shell vacancy PENELOPE
    assigns is not modelled (Beam Weaver does not model atomic relaxation).
    """
    if E < PAIR_THRESHOLD_MEV:
        return (old_dir, E, [], "pair_subthresh", [0] * N_SHELLS)
    eps = _pair_sample_eps(E)             # electron TOTAL-energy fraction
    eps_e = max(eps * E - mec2, 0.0)          # E_- = εE − me·c²   (kinetic)
    eps_p = max((1.0 - eps) * E - mec2, 0.0)  # E_+ = (1−ε)E − me·c² (kinetic)
    theta_e = math.acos(max(-1.0, min(1.0, _pair_lepton_costheta(eps_e))))
    theta_p = math.acos(max(-1.0, min(1.0, _pair_lepton_costheta(eps_p))))
    # Azimuths sampled INDEPENDENTLY and uniformly — PENELOPE-2018 §2.4.1.1
    # verbatim ('the azimuthal angles φ− and φ+ ... are sampled independently
    # and uniformly'); the nucleus absorbs the transverse recoil.  This also
    # matches the independent electron and positron direction heads.
    phi_e = 2 * math.pi * random.random()
    phi_p = 2 * math.pi * random.random()
    dir_e = rotate_direction(old_dir, theta_e, phi_e)
    dir_p = rotate_direction(old_dir, theta_p, phi_p)
    secs = [("electron", eps_e, dir_e, "pair_e"),
            ("positron", eps_p, dir_p, "pair_p")]
    # Default shell onehot (no photoelectric event)
    shell_onehot = [0] * N_SHELLS
    return (np.array([0, 0, 0]), 0.0, secs, "pair", shell_onehot)


def sample_photon_interaction(E, direction, data:WaterPhotonData):
    (coh,inc,pho,ppr,tot)= data.partial_cs(E)
    if tot<ATTENUATION_FLOOR_CM_INV:
        return (direction,E,[],"none", [0] * N_SHELLS)
    r= random.random()*tot
    if r<coh:
        return sample_rayleigh_event(E,direction,data)
    r-=coh
    if r<inc:
        return sample_compton_event(E,direction,data)
    r-=inc
    if r<pho:
        return sample_photoelectric_event(E,direction,data)
    return sample_pair_event(E,direction,data)


def load_stopping_power(csv_path=ELECTRON_STOPPING_POWER_TABLE):
    """Load both collisional and radiative stopping powers"""
    df = pd.read_csv(csv_path)
    return (
        df["E_MeV"].values,          # Energy grid (MeV)
        df["S_col_MeV_per_cm"].values,   # Collisional stopping power
        df["S_rad_MeV_per_cm"].values    # Radiative stopping power
    )


def stopping_power(E, Egrid, S_col_vals, S_rad_vals):
    """Return interpolated collisional and radiative stopping powers"""
    # Linear interpolation in log-log space (common for stopping powers)
    log_E = np.log(np.clip(E, 1e-6, None))
    log_S_col = np.interp(log_E, np.log(Egrid), np.log(S_col_vals))
    log_S_rad = np.interp(log_E, np.log(Egrid), np.log(S_rad_vals))
    return np.exp(log_S_col), np.exp(log_S_rad)


def transport_lepton_csda(E_lepton, direction, start_pos, dose_tally, env,
                          Egrid=None, S_col_vals=None,
                          S_rad_vals=None, is_positron=False,
                          ecut=DEFAULT_TRANSPORT_ECUT_MEV):
    """
    Shared continuous-slowing-down transport for electrons and positrons.
    Energies are in MeV, lengths in cm, and stopping powers in MeV/cm.

    The kinetic-energy ledger checks
        E_lepton = col_deposit + cutoff_deposit + soft_rad_deposit
                   + brems_energy + escaped_kinetic.

    Steps are clipped at the phantom boundary and the remaining kinetic
    energy escapes. Subcutoff residual energy deposits locally when the
    track stops inside the phantom. An absorbed positron annihilates at rest;
    an escaping positron does not create annihilation photons in the phantom.

    Bremsstrahlung uses a simplified budget-conserving 1/W spectrum with
    emission positions distributed along the in-phantom track. Residual soft
    radiative energy deposits locally. Positrons use the electron stopping
    powers, and in-flight annihilation is omitted. These approximations need
    separate validation against a trusted charged-particle transport code.

    Returns (total_local_deposit, secondaries, ledger).  Photon secondaries:
        ("photon", E_MeV, dir_ndarray, tag, (x, y, z))
    """
    dirn = np.asarray(direction, dtype=float)
    nrm = np.linalg.norm(dirn)
    dirn = dirn / nrm if nrm > 1e-14 else np.array([0.0, 0.0, 1.0])
    pos = np.array(start_pos, dtype=float)
    p0 = pos.copy()
    ledger = {"E_in": float(E_lepton), "col_deposit": 0.0, "cutoff_deposit": 0.0,
              "soft_rad_deposit": 0.0, "brems_energy": 0.0,
              "escaped_kinetic": 0.0, "annih_energy": 0.0,
              "annih_potential_escaped": 0.0, "n_brems_photons": 0,
              "escaped": False, "species": "positron" if is_positron else "electron"}
    secondaries = []

    def _inside(p):
        return (env.xmin <= p[0] <= env.xmax and env.ymin <= p[1] <= env.ymax
                and env.zmin <= p[2] <= env.zmax)

    def _exit_distance(p, d):
        # Distance to the first axis-aligned boundary.
        vals = []
        for xi, ui, lo, hi in ((p[0], d[0], env.xmin, env.xmax),
                               (p[1], d[1], env.ymin, env.ymax),
                               (p[2], d[2], env.zmin, env.zmax)):
            if ui > 1e-15:
                vals.append((hi - xi) / ui)
            elif ui < -1e-15:
                vals.append((lo - xi) / ui)
        vals = [v for v in vals if v >= 0.0]
        return min(vals) if vals else float("inf")

    def _tally(p, dE):
        k = int((p[2] - env.zmin) / env.dz)
        if 0 <= k < env.pdd_bins:
            dose_tally[k] += dE

    remaining = float(E_lepton)
    rad_budget = 0.0
    if not _inside(pos):
        # born outside (e.g. numerically on the boundary): everything escapes
        ledger["escaped"] = True
        ledger["escaped_kinetic"] = remaining
        if is_positron:
            ledger["annih_potential_escaped"] = 2.0 * mec2
        return 0.0, secondaries, ledger

    inside_end = pos.copy()
    while remaining > ecut:
        S_col, S_rad = stopping_power(remaining, Egrid, S_col_vals, S_rad_vals)
        S_total = S_col + S_rad
        if S_total <= 0:
            break
        requested = max(0.001, 0.0001 / S_total)      # 0.001 cm minimum step
        # Clip the step to the exact box exit so the
        # in-phantom fraction of a boundary-crossing step is DEPOSITED before
        # the remainder escapes.
        to_exit = _exit_distance(pos, dirn)
        step = min(requested, max(to_exit, 0.0))
        if step <= 1e-15:                             # 3d: on the boundary
            ledger["escaped"] = True
            ledger["escaped_kinetic"] = remaining
            break
        dE = min(S_total * step, remaining)           # 3a: clamp to remaining
        dE_col = dE * (S_col / S_total)
        dE_rad = dE - dE_col
        midpoint = pos + 0.5 * step * dirn            # midpoint tally bin
        pos = pos + step * dirn
        inside_end = pos.copy()
        _tally(midpoint, dE_col)
        ledger["col_deposit"] += dE_col
        rad_budget += dE_rad
        remaining -= dE
        # Reaching the boundary terminates the track even below the cutoff.
        # in the boundary voxel that had physically left the phantom (the
        # ledger still closed, which is why the closure gates missed it).
        if step >= to_exit - 1e-12 and remaining > 0.0:    # 3d: exact exit
            ledger["escaped"] = True
            ledger["escaped_kinetic"] = remaining
            break

    if not ledger["escaped"] and remaining > 0.0:
        # 3c: deposit the sub-cutoff residual locally (absorption energy)
        _tally(inside_end, remaining)
        ledger["cutoff_deposit"] = remaining
        remaining = 0.0

    # 3b: budget-conserving bremsstrahlung from the in-phantom radiative budget
    budget = rad_budget
    Wcap = max(min(float(E_lepton), budget), ecut)
    while budget > ecut:
        W = ecut * (Wcap / ecut) ** random.random()   # 1/W spectrum on (ecut, Wcap)
        W = min(W, budget)
        if W <= ecut:
            break
        upos = p0 + random.random() * (inside_end - p0)
        theta = math.acos(1 - random.random() ** 0.5) # forward-peaked (prev. law)
        phi = 2 * math.pi * random.random()
        pdir = rotate_direction(dirn, theta, phi)
        secondaries.append(("photon", W, pdir, "brem",
                            (upos[0], upos[1], upos[2])))
        ledger["brems_energy"] += W
        ledger["n_brems_photons"] += 1
        budget -= W
    if budget > 0.0:                                  # soft remainder: local
        _tally(inside_end, budget)
        ledger["soft_rad_deposit"] = budget

    if is_positron:
        if ledger["escaped"]:
            ledger["annih_potential_escaped"] = 2.0 * mec2
        else:
            # annihilation at rest AFTER absorption (3c ordering):
            # two photons of exactly mec2, back-to-back, isotropic orientation
            cost = 2.0 * random.random() - 1.0
            sint = math.sqrt(max(0.0, 1.0 - cost * cost))
            aphi = 2.0 * math.pi * random.random()
            d = np.array([sint * math.cos(aphi), sint * math.sin(aphi), cost])
            p = (inside_end[0], inside_end[1], inside_end[2])
            secondaries.append(("photon", mec2,  d, "annih", p))
            secondaries.append(("photon", mec2, -d, "annih", p))
            ledger["annih_energy"] = 2.0 * mec2

    total_local = (ledger["col_deposit"] + ledger["cutoff_deposit"]
                   + ledger["soft_rad_deposit"])
    _closure = (total_local + ledger["brems_energy"]
                + ledger["escaped_kinetic"]) - float(E_lepton)
    ledger["closure_err"] = abs(_closure)
    # A broken kinetic-energy ledger
    # aborts immediately instead of surfacing later as a dose anomaly.
    if ledger["closure_err"] > 1e-9 * max(1.0, float(E_lepton)):
        raise AssertionError(
            f"lepton ledger closure failed: {_closure:.3e} MeV")
    return total_local, secondaries, ledger
