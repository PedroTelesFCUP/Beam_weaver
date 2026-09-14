"""Beam Weaver: validation."""

# Copyright (c) 2025–2026 Pedro Teles. Apache-2.0.

import json
import math
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from .constants import PAIR_THRESHOLD_MEV
from .constants import (
    BW_VERSION, mec2, PHOTO_SHELL_BINDINGS_MEV, _PAIR_R_EQ, _PAIR_Z_EQ,
    DEFAULT_FACTOR_VALIDATION_SAMPLES, DEFAULT_REFERENCE_VALIDATION_SAMPLES,
    KEV_PER_MEV, N_SHELLS, PROCESS_INDEX, SHELL_INDEX, REFERENCE_POLAR_GRID_POINTS,
)
from .coordinates import norm_logE
from .dataset import (
    deterministic_seed, sample_compton_training_data, sample_pair_training_data, sample_photoelectric_training_data,
    sample_process_counts, sample_rayleigh_training_data, sample_shell_counts,
)
from .physics import ComptonEnergyTransferSampler
from .physics import (
    _pair_F0, _pair_fC, _sauter_dcs_shape,
    sample_compton_event, sample_pair_event, sample_photoelectric_event, sample_rayleigh_event,
)


def compton_angular_distribution(E_in, sampler):
    """Normalized Klein–Nishina × S(q) theta density on 180 inclusive degree points.

    E_in is in MeV. The Jacobian sin(theta) converts the solid-angle
    density to a polar-angle density, matching the reference sampler.
    """
    alpha = E_in / mec2

    angles_deg = np.linspace(0, 180, REFERENCE_POLAR_GRID_POINTS)
    angles_rad = np.radians(angles_deg)
    c_grid = np.cos(angles_rad)
    eps_g  = 1.0 / (1.0 + alpha * (1.0 - c_grid))          # KN-linked
    sin2_g = np.clip(1.0 - c_grid**2, 0.0, None)
    kn_g   = eps_g**2 * (eps_g + 1.0 / eps_g - sin2_g)     # dσ/dΩ shape
    q_g    = sampler.compute_q(eps_g, c_grid)
    S_g    = np.interp(q_g / 2, sampler.q_grid, sampler.S_grid,
                       left=0.0, right=sampler.S_max) / sampler.S_max
    dist = np.clip(kn_g * S_g, 0.0, None) * np.sin(angles_rad)
    s = dist.sum()
    if s > 0:
        dist = dist / s
    return dist


def rayleigh_angular_distribution(E_in, data):
    """Normalized (1 + cos²(theta)) F(q)² sin(theta) on 180 degree points.

    E_in is in MeV; F(q) is the same material form factor used by the
    reference Rayleigh sampler. This grid includes 0 and 180 degrees.
    """
    k     = (E_in*KEV_PER_MEV) / data.HC_KEV_A
    angles_deg = np.linspace(0, 180, REFERENCE_POLAR_GRID_POINTS)
    angles_rad = np.radians(angles_deg)
    cos_values = np.cos(angles_rad)
    dist = np.zeros_like(angles_deg)
    
    for i, (angle_rad, cos_val) in enumerate(zip(angles_rad, cos_values)):
        q_i = 2 * k * math.sin(angle_rad/2)
        F_q_i = data.coherent_form_factor(q_i)
        ratio_i = (F_q_i / data.F0)**2
        dist[i] = ratio_i * (1 + cos_val**2)/2 * np.sin(angle_rad)
    
    # Normalize
    if np.sum(dist) > 0:
        dist = dist / np.sum(dist)
    return dist


def photo_angular_distribution(E_in, shell="O_K"):
    """Normalized photoelectron polar density on 180 inclusive degree points.

    K shells use the shared Sauter shape; L shells are isotropic in
    solid angle. E_in and derived electron kinetic energy are in MeV.
    """
    angles_deg = np.linspace(0, 180, REFERENCE_POLAR_GRID_POINTS)
    angles_rad = np.radians(angles_deg)
    dist = np.ones(REFERENCE_POLAR_GRID_POINTS)

    # Isotropic solid-angle sampling gives a sin(theta) polar density.
    if shell in ('O_L1', 'O_L2', 'O_L3'):
        for i, angle_rad in enumerate(angles_rad):
            dist[i] = np.sin(angle_rad)
        # Normalize
        if np.sum(dist) > 0:
            dist = dist / np.sum(dist)
        return dist

    Eb_MeV = PHOTO_SHELL_BINDINGS_MEV[shell]
    T_MeV  = E_in - Eb_MeV                # electron kinetic energy, MeV
    if T_MeV <= 0.0:
        return np.zeros(REFERENCE_POLAR_GRID_POINTS)  # photon below binding energy

    gamma = 1.0 + T_MeV / mec2
    beta  = math.sqrt(T_MeV * (T_MeV + 2.0 * mec2)) / (T_MeV + mec2)

    dist = _sauter_dcs_shape(np.cos(angles_rad), beta, gamma) * np.sin(angles_rad)
    s = dist.sum()
    if s > 0:
        dist = dist / s
    return dist


def pair_angular_distribution(E_in):
    """Pair-electron polar marginal on 180 inclusive degree points.

    Integrate p(epsilon | E) (1-beta²)/(2(1-beta cos(theta))²) sin(theta)
    over the electron total-energy fraction using the reference pair
    sampler's energy-fraction density. E_in is in MeV. The quadrature
    and its near-threshold fallback retain the reference implementation.
    """
    if E_in < PAIR_THRESHOLD_MEV:
        return np.zeros(REFERENCE_POLAR_GRID_POINTS)
    kappa = E_in / mec2
    inv_k = 1.0 / kappa
    g0 = (4.0 * math.log(_PAIR_R_EQ) - 4.0 * _pair_fC(_PAIR_Z_EQ)
          + _pair_F0(kappa, _PAIR_Z_EQ))
    eg = np.linspace(inv_k, 1.0 - inv_k, 400)[1:-1]          # ε quadrature grid
    b = _PAIR_R_EQ * (1.0 / (2.0 * kappa)) / (eg * (1.0 - eg))
    atanb = np.arctan(1.0 / b)
    l1 = np.log(1.0 + b * b)
    core = 4.0 - 4.0 * b * atanb - 3.0 * np.log(1.0 + 1.0 / (b * b))
    phi1 = np.clip(7.0/3.0 - 2.0*l1 - 6.0*b*atanb - b*b*core + g0, 0.0, None)
    phi2 = np.clip(11.0/6.0 - 2.0*l1 - 3.0*b*atanb + 0.5*b*b*core + g0, 0.0, None)
    pe = 2.0 * (0.5 - eg) ** 2 * phi1 + phi2                 # energy-fraction density shape
    s = pe.sum()
    if s <= 0.0:                                             # near-threshold clip
        pe = np.ones_like(eg)
        s = pe.sum()
    pe = pe / s
    T = np.clip(eg * E_in - mec2, 0.0, None)                 # electron KINETIC
    beta = np.sqrt(T * (T + 2.0 * mec2)) / (T + mec2)        # β(ε)
    one_m_b2 = 1.0 - beta ** 2

    angles_deg = np.linspace(0, 180, REFERENCE_POLAR_GRID_POINTS)
    angles_rad = np.radians(angles_deg)
    mu = np.cos(angles_rad)[:, None]                          # (180, 1)
    dens = (pe[None, :] * one_m_b2[None, :]
            / (2.0 * (1.0 - beta[None, :] * mu) ** 2)).sum(axis=1)
    dist = dens * np.sin(angles_rad)                          # θ-space density
    ssum = dist.sum()
    if ssum > 0:
        dist = dist / ssum
    return dist


def _head_probs(policy, head, E, sh=None, f=None):
    """Return one head's probabilities and bin edges (None for categories)."""
    import torch

    dev = policy._dev()
    xE = policy._energy_features(E)
    with torch.no_grad():
        if head == "process":
            lg = policy.process_head(xE)[0].clone()
            if E <= PAIR_THRESHOLD_MEV:
                lg[PROCESS_INDEX["pair"]] = float("-inf")
            return torch.softmax(lg, -1).cpu().numpy(), None
        if head == "shell":
            return torch.softmax(policy.shell_head(xE)[0],
                                 -1).cpu().numpy(), None
        if head == "ray_q2":
            return torch.softmax(policy.rayleigh_q2_head(xE)[0],
                                 -1).cpu().numpy(), \
                policy.edge_arrays["ray_q2"]
        if head == "comp_u":
            return torch.softmax(policy.compton_u_head(xE)[0],
                                 -1).cpu().numpy(), \
                policy.edge_arrays["comp_u"]
        if head == "photo_nu":
            soh = torch.zeros(1, N_SHELLS, device=dev); soh[0, sh] = 1.0
            lg = policy.photo_nu_head(torch.cat([xE, soh], 1))[0]
            return torch.softmax(lg, -1).cpu().numpy(), \
                policy.edge_arrays["photo_nu"]
        if head == "pair_frac":
            return torch.softmax(policy.pair_fraction_head(xE)[0],
                                 -1).cpu().numpy(), \
                policy.edge_arrays["pair_frac"]
        if head in ("ray_phi", "comp_phi"):
            hd = (policy.rayleigh_phi_head if head == "ray_phi"
                  else policy.compton_phi_head)
            return torch.softmax(hd(1)[0], -1).cpu().numpy(), \
                policy.edge_arrays["phi"]
        if head == "photo_phi":
            soh = torch.zeros(1, N_SHELLS, device=dev); soh[0, sh] = 1.0
            return torch.softmax(policy.photo_phi_head(soh)[0],
                                 -1).cpu().numpy(), \
                policy.edge_arrays["phi"]
        # pair directional heads at conditioning (E, f)
        x = torch.tensor([[float(norm_logE(E)),
                           float(f if "_e_" in head else 1.0 - f)]],
                         dtype=torch.float32, device=dev)
        hd = {"pair_e_polar": policy.pair_electron_polar_head,
              "pair_e_phi": policy.pair_electron_phi_head,
              "pair_p_polar": policy.pair_positron_polar_head,
              "pair_p_phi": policy.pair_positron_phi_head}[head]
        ed = (policy.edge_arrays[f"pair_nu_{'e' if '_e_' in head else 'p'}"]
              if "polar" in head else policy.edge_arrays["phi"])
        return torch.softmax(hd(x)[0], -1).cpu().numpy(), ed


def _head_sample(probs, edges, n, rng):
    """Draw categorical bins and a uniform position within each bin.

    KS is invariant under strictly monotone coordinate transforms.
    """
    k = rng.choice(len(probs), size=n, p=probs / probs.sum())
    a, b = edges[:-1][k], edges[1:][k]
    return a + rng.random(n) * (b - a)


def validate_policy(policy, data, energies=None, M=DEFAULT_FACTOR_VALIDATION_SAMPLES,
                         out_json="v040_local_validation_report.json",
                         figures=True, verbose=True, output_dir=None):
    """Compare every learned head with independent local MC samples.

    Relative report and figure paths are resolved under output_dir when
    supplied. An absolute out_json path remains explicit.
    """
    output_root = Path(output_dir) if output_dir is not None else Path(".")
    output_root.mkdir(parents=True, exist_ok=True)
    out_json = Path(out_json)
    if not out_json.is_absolute():
        out_json = output_root / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    energies = energies or [0.02, 0.05, 0.1, 0.5, 1.05, 2.0, 5.0, 10.0]
    thr = PAIR_THRESHOLD_MEV
    figdir = output_root / "v040_local_validation"
    if figures:
        os.makedirs(figdir, exist_ok=True)
    rep = {"M": M, "energies": energies, "heads": {}}
    Kphi = len(policy.edge_arrays["phi"]) - 1
    mc_cache = {}

    def _mc(factor, E, sh, tag):
        key = (factor, E, sh, tag)
        if key not in mc_cache:
            seed = deterministic_seed("lv" + tag, factor, f"{E}|{sh}")
            fn = {"process": lambda: sample_process_counts(E, M, data, seed),
                  "shell": lambda: sample_shell_counts(E, M, data, seed),
                  "ray": lambda: sample_rayleigh_training_data(E, M, data, seed),
                  "comp": lambda: sample_compton_training_data(E, M, data, seed),
                  "photo": lambda: sample_photoelectric_training_data(E, sh, M, data, seed),
                  "pair": lambda: sample_pair_training_data(E, M, data, seed)}[factor]
            mc_cache[key] = fn()
        return mc_cache[key]

    def _hist(x, edges):
        h = np.histogram(x, bins=edges)[0].astype(float)
        return h / max(h.sum(), 1)

    def _row(head, E, sh, q, p1, p2, x_mc=None, edges=None, extra=None):
        tv = 0.5 * float(np.abs(q - p1).sum())
        floor = 0.5 * float(np.abs(p1 - p2).sum())
        kl = float(np.sum(p1[p1 > 0]
                          * np.log(p1[p1 > 0]
                                   / np.maximum(q[p1 > 0], 1e-12))))
        ks = None
        if x_mc is not None and edges is not None:
            rng = np.random.default_rng(deterministic_seed("ks", head, f"{E}|{sh}"))
            d = _head_sample(q, edges, min(len(x_mc), 4096), rng)
            a1 = np.sort(np.asarray(x_mc)[:len(d)]); b1 = np.sort(d)
            grid = np.union1d(a1, b1)
            ks = float(np.max(np.abs(
                np.searchsorted(a1, grid, "right") / len(a1)
                - np.searchsorted(b1, grid, "right") / len(b1))))
        r = {"E": E, "shell": sh, "TV_vs_MC": tv, "KL": kl,
             "KS_physical": ks, "MC_replication_TV": floor}
        if extra:
            r.update(extra)
        return r

    def _fig(head, E, sh, edges, p1, q):
        if not figures:
            return
        c = 0.5 * (edges[:-1] + edges[1:])
        plt.figure(figsize=(5, 3))
        plt.step(c, p1, where="mid", label="MC")
        plt.step(c, q, where="mid", label="policy")
        plt.title(f"{head} E={E}" + (f" H={sh}" if sh is not None else ""))
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{figdir}/{head}_E{E}"
                    + (f"_H{sh}" if sh is not None else "") + ".png",
                    dpi=110)
        plt.close()

    for head in ("process", "shell", "ray_q2", "ray_phi", "comp_u",
                 "comp_phi", "photo_nu", "photo_phi", "pair_frac",
                 "pair_e_polar", "pair_e_phi", "pair_p_polar",
                 "pair_p_phi"):
        rows = []
        pairlike = head.startswith("pair")
        for E in energies:
            if pairlike and E <= thr:
                continue
            shells = range(N_SHELLS) if head in ("photo_nu", "photo_phi") \
                else [None]
            for sh in shells:
                if head in ("process", "shell"):
                    fac = "process" if head == "process" else "shell"
                    q, _ = _head_probs(policy, head, E)
                    c1 = _mc(fac, E, None, "1")["counts"].astype(float)
                    c2 = _mc(fac, E, None, "2")["counts"].astype(float)
                    rows.append(_row(head, E, sh, q, c1 / c1.sum(),
                                     c2 / c2.sum()))
                elif head in ("ray_q2", "comp_u", "photo_nu",
                              "pair_frac"):
                    fac, fld = {"ray_q2": ("ray", "lq2r"),
                                "comp_u": ("comp", "u"),
                                "photo_nu": ("photo", "nu"),
                                "pair_frac": ("pair", "f")}[head]
                    q, e = _head_probs(policy, head, E, sh)
                    g1 = _mc(fac, E, sh, "1"); g2 = _mc(fac, E, sh, "2")
                    p1 = _hist(g1[fld], e)
                    p2 = _hist(g2[fld], e)
                    rows.append(_row(head, E, sh, q, p1, p2,
                                     x_mc=g1[fld], edges=e))
                    if sh in (None, SHELL_INDEX["O_K"]) and E in (0.05, 1.05, 5.0):
                        _fig(head, E, sh, e, p1, q)
                elif head in ("ray_phi", "comp_phi", "photo_phi"):
                    fac = {"ray_phi": "ray", "comp_phi": "comp",
                           "photo_phi": "photo"}[head]
                    q, e = _head_probs(policy, head, E, sh)
                    g1 = _mc(fac, E, sh, "1"); g2 = _mc(fac, E, sh, "2")
                    ok1 = g1["phi_valid"].astype(bool)
                    ok2 = g2["phi_valid"].astype(bool)
                    p1 = _hist(g1["phi"][ok1], e)
                    p2 = _hist(g2["phi"][ok2], e)
                    uni = 0.5 * float(np.abs(q - 1.0 / Kphi).sum())
                    rows.append(_row(head, E, sh, q, p1, p2,
                                     extra={"TV_vs_uniform": uni}))
                else:                       # pair directional heads
                    g1 = _mc("pair", E, None, "1")
                    g2 = _mc("pair", E, None, "2")
                    lep = "e" if "_e_" in head else "p"
                    fl1 = g1["f"] if lep == "e" else 1.0 - g1["f"]
                    fl2 = g2["f"] if lep == "e" else 1.0 - g2["f"]
                    for f0 in (0.25, 0.5, 0.75):
                        w1 = np.abs(fl1 - f0) < 0.125
                        w2 = np.abs(fl2 - f0) < 0.125
                        if w1.sum() < 64 or w2.sum() < 64:
                            continue
                        q, e = _head_probs(policy, head, E,
                                                  f=(f0 if lep == "e"
                                                     else 1.0 - f0))
                        if "polar" in head:
                            v1 = g1[f"nu_{lep}"][w1]
                            v2 = g2[f"nu_{lep}"][w2]
                            p1 = _hist(v1, e); p2 = _hist(v2, e)
                            rows.append(_row(head, E, None, q, p1, p2,
                                             x_mc=v1, edges=e,
                                             extra={"f_window": f0}))
                        else:
                            ok1 = g1[f"phi_{lep}_valid"].astype(bool) & w1
                            ok2 = g2[f"phi_{lep}_valid"].astype(bool) & w2
                            p1 = _hist(g1[f"phi_{lep}"][ok1], e)
                            p2 = _hist(g2[f"phi_{lep}"][ok2], e)
                            rows.append(_row(head, E, None, q, p1, p2,
                                             extra={"f_window": f0}))
        if not rows:
            continue
        agg = {"worst_TV": max(r["TV_vs_MC"] for r in rows),
               "mean_TV": float(np.mean([r["TV_vs_MC"] for r in rows])),
               "mean_replication_TV": float(np.mean(
                   [r["MC_replication_TV"] for r in rows])),
               "n_rows": len(rows)}
        rep["heads"][head] = {"aggregate": agg, "rows": rows}
        if verbose:
            print(f"  [v4:val] {head:13s} meanTV={agg['mean_TV']:.4f} "
                  f"worstTV={agg['worst_TV']:.4f} "
                  f"(floor {agg['mean_replication_TV']:.4f}, "
                  f"{agg['n_rows']} rows)")
    with open(out_json, "w") as fh:
        json.dump(rep, fh, indent=2)
    if verbose:
        print(f"  [v4:val] all 13 heads validated -> {out_json}"
              + (f"; figures -> {figdir}/" if figures else ""))
    return rep


def validate_reference_samplers(data, n_samples=DEFAULT_REFERENCE_VALIDATION_SAMPLES,
                               ks_gate=0.03):
    """Explicit reference-sampler check against angular target distributions.

    This is an optional validation operation, not a startup or training
    prerequisite. The binned KS-like statistic retains the existing grid,
    energies and gate; it is not a test of external transport accuracy.
    """
    if n_samples <= 0 or not 0 < ks_gate < 1:
        raise ValueError("n_samples must be positive and ks_gate must be in (0, 1)")
    t0 = time.time()
    old_dir = np.array([0.0, 0.0, 1.0])
    failures = []

    def _ks_hist_vs_target(thetas_rad, target_dist):
        h, edges = np.histogram(thetas_rad, bins=180, range=(0.0, math.pi))
        # Reference densities use inclusive degree points. Interpolate to
        # histogram bin centres to avoid a half-bin registration shift.
        cen_deg = np.degrees(0.5 * (edges[1:] + edges[:-1]))
        t = np.asarray(target_dist, dtype=float)
        tgrid = np.linspace(0.0, 180.0, t.size)
        q = np.clip(np.interp(cen_deg, tgrid, t), 0.0, None)
        p = h.astype(float);  p /= max(p.sum(), 1.0)
        q = q / max(q.sum(), 1e-30)
        return float(np.abs(np.cumsum(p) - np.cumsum(q)).max())

    def _theta_from_dir(new_dir):
        return math.acos(float(np.clip(np.dot(old_dir, new_dir), -1.0, 1.0)))

    print(f"── v{BW_VERSION} reference-sampler validation: {n_samples:,} samples/channel/energy, "
          f"gate KS < {ks_gate} ──")

    # 1) Rayleigh — q²-CDF sampler vs form-factor kernel
    for E in (0.010, 0.030, 0.060, 0.100):
        th = [ _theta_from_dir(sample_rayleigh_event(E, old_dir, data)[0])
               for _ in range(n_samples) ]
        tdist = rayleigh_angular_distribution(E, data)
        k = _ks_hist_vs_target(th, tdist)
        st = "OK " if k < ks_gate else "FAIL"
        print(f"  [rayleigh] E={E*KEV_PER_MEV:6.1f} keV  KS={k:.4f}  {st}")
        if k >= ks_gate:
            failures.append(f"rayleigh @ {E} MeV KS={k:.4f}")

    # One reusable composition sampler per incident energy.
    for E in (0.020, 0.050, 0.100, 0.500, 1.000):
        spl = ComptonEnergyTransferSampler(E)
        th = [ _theta_from_dir(sample_compton_event(E, old_dir, sampler=spl)[0])
               for _ in range(n_samples) ]
        tdist = compton_angular_distribution(E, spl)
        k = _ks_hist_vs_target(th, tdist)
        st = "OK " if k < ks_gate else "FAIL"
        print(f"  [compton ] E={E*KEV_PER_MEV:6.1f} keV  KS={k:.4f}  {st}")
        if k >= ks_gate:
            failures.append(f"compton @ {E} MeV KS={k:.4f}")

    # Force each selected shell through the complete photoelectric sampler
    # to exercise rejection sampling and angle reconstruction.
    class _ForceShell:
        def __init__(self, idx): self.idx = idx
        def sample_photo_shell_index(self, E): return self.idx
    for shell_idx, shell_name, energies in (
            (SHELL_INDEX["O_K"], "O_K", (0.0011, 0.005, 0.020, 0.100, 0.500)),
            (SHELL_INDEX["O_L1"], "O_L1", (0.002,))):
        stub = _ForceShell(shell_idx)
        for E in energies:
            th = []
            for _ in range(n_samples):
                r = sample_photoelectric_event(E, old_dir, stub)
                sec = r[2]
                th.append(_theta_from_dir(sec[0][2]))
            tdist = photo_angular_distribution(E, shell=shell_name)
            k = _ks_hist_vs_target(th, tdist)
            st = "OK " if k < ks_gate else "FAIL"
            print(f"  [photo:{shell_name:4s}] E={E*KEV_PER_MEV:6.1f} keV  KS={k:.4f}  {st}")
            if k >= ks_gate:
                failures.append(f"photo {shell_name} @ {E} MeV KS={k:.4f}")

    # Pair-electron marginal integrated over sampled energy fractions.
    for E in (2.0, 5.0):
        th = []
        for _ in range(n_samples):
            r = sample_pair_event(E, old_dir, data)
            th.append(_theta_from_dir(r[2][0][2]))    # electron direction
        tdist = pair_angular_distribution(E)
        k = _ks_hist_vs_target(th, tdist)
        st = "OK " if k < ks_gate else "FAIL"
        print(f"  [pair    ] E={E*KEV_PER_MEV:6.1f} keV  KS={k:.4f}  {st}")
        if k >= ks_gate:
            failures.append(f"pair @ {E} MeV KS={k:.4f}")

    dt = time.time() - t0
    if failures:
        raise RuntimeError(
            f"reference-sampler validation failed ({dt:.0f}s): " + "; ".join(failures))
    print(f"Reference samplers passed: all channels within KS < {ks_gate} ({dt:.0f}s)")
    return {"passed": True, "n_samples": n_samples, "ks_gate": ks_gate,
            "elapsed_seconds": dt}
