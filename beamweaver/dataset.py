"""Beam Weaver: dataset."""

# Copyright (c) 2025–2026 Pedro Teles & João Melo. Apache-2.0.

from .constants import runtime_physics_metadata
from .constants import PAIR_THRESHOLD_MEV
from .constants import BW4_COORDS
from .constants import BW4_EMAX_MEV
from .constants import BW4_EMIN_MEV
from .constants import BW4_K
from .constants import BW4_M_DEFAULT
from .constants import BW4_NORM_LOGE_SPEC
from .constants import BW4_PAIR_TRAIN_EXTRA
from .constants import BW4_PHI_EPS
from .constants import PROCESS_NAMES
from .constants import BW4_SANITY_FLOOR
from .constants import SHELL_NAMES
from .constants import BW4_S_SUB
from .constants import BW4_TEST_E
from .constants import (
    BW_VERSION, N_SHELLS,
    PHOTON_CROSS_SECTION_TABLE, RAYLEIGH_CROSS_SECTION_TABLE,
    PHOTO_SHELL_TABLE, INCOHERENT_SCATTERING_TABLE,
    COHERENT_FORM_FACTOR_TABLE, ELECTRON_STOPPING_POWER_TABLE,
)
from .physics import ComptonEnergyTransferSampler
_BW4_NULLC = {}  # Computed values cached for this module.
from .constants import _BW4_Z
import hashlib
import json
import os
from .coordinates import lq2r_from_cos
from .coordinates import q2_from_cos
from .coordinates import u_from_tau
import math
from pathlib import Path
import numpy as np
import random
from .physics import sample_compton_event
from .physics import sample_pair_event
from .physics import sample_photoelectric_event_for_shell
from .physics import sample_rayleigh_event


def build_energy_grids():
    E_tr = np.geomspace(BW4_EMIN_MEV, BW4_EMAX_MEV, 64)
    E_va = np.sqrt(E_tr[:-1] * E_tr[1:])
    E_te = np.array(sorted(BW4_TEST_E))
    thr = PAIR_THRESHOLD_MEV
    g = {"train": np.unique(np.concatenate(
             [E_tr, np.array(BW4_PAIR_TRAIN_EXTRA)])),
         "val": E_va, "test": E_te, "pair_threshold": thr}
    h = hashlib.sha256()
    for k in ("train", "val", "test"):
        h.update(k.encode()); h.update(g[k].tobytes())
    g["sha256"] = h.hexdigest()
    return g


def deterministic_seed(role, factor, tag):
    d = hashlib.sha256(f"v040|{role}|{factor}|{tag}".encode()).digest()
    return int.from_bytes(d[:4], "big")


def _direction_to_angles(d):
    w = max(-1.0, min(1.0, float(d[2])))
    return math.acos(w), math.atan2(float(d[1]), float(d[0])) % (2 * math.pi)


def sample_process_counts(E, M, data, seed):
    np.random.seed(seed % 2**32)
    coh, inc, pho, ppr, _ = data.partial_cs(E)
    p = np.array([coh, inc, pho, ppr], np.float64); p /= p.sum()
    return {"counts": np.random.multinomial(M, p).astype(np.int64)}


def sample_shell_counts(E, M, data, seed):
    np.random.seed(seed % 2**32)
    p = np.asarray(data.water_shell_data.shell_probs(E), np.float64)
    p /= p.sum()
    return {"counts": np.random.multinomial(M, p).astype(np.int64)}


def sample_rayleigh_training_data(E, M, data, seed):
    random.seed(seed); np.random.seed(seed % 2**32)
    cos = np.empty(M); phi = np.empty(M)
    for i in range(M):
        nd, _, _, _, _ = sample_rayleigh_event(E, _BW4_Z, data)
        t, p = _direction_to_angles(nd); cos[i] = math.cos(t); phi[i] = p
    pv = (1 - cos**2) > BW4_PHI_EPS**2
    return {"q2": q2_from_cos(cos, E),
            "lq2r": lq2r_from_cos(cos), "phi": phi,
            "phi_valid": pv.astype(np.uint8)}


def sample_compton_training_data(E, M, data, seed):
    random.seed(seed); np.random.seed(seed % 2**32)
    comp = ComptonEnergyTransferSampler(E)
    u = np.empty(M); cos = np.empty(M); phi = np.empty(M)
    for i in range(M):
        nd, Eout, _, _, _ = sample_compton_event(E, _BW4_Z, data, sampler=comp)
        t, p = _direction_to_angles(nd)
        u[i] = u_from_tau(Eout / E, E); cos[i] = math.cos(t); phi[i] = p
    pv = (1 - cos**2) > BW4_PHI_EPS**2
    return {"u": np.clip(u, 0.0, 1.0), "phi": phi,
            "phi_valid": pv.astype(np.uint8)}


def sample_photoelectric_training_data(E, sh, M, data, seed):
    random.seed(seed); np.random.seed(seed % 2**32)
    cos = np.empty(M); phi = np.empty(M)
    for i in range(M):
        _, _, secs, tag, _ = sample_photoelectric_event_for_shell(E, _BW4_Z, data, sh)
        assert tag == "photo"
        t, p = _direction_to_angles(secs[0][2]); cos[i] = math.cos(t); phi[i] = p
    pv = (1 - cos**2) > BW4_PHI_EPS**2
    return {"nu": 1.0 - cos, "phi": phi, "phi_valid": pv.astype(np.uint8)}


def sample_pair_training_data(E, M, data, seed):
    random.seed(seed); np.random.seed(seed % 2**32)
    f = np.empty(M); ne = np.empty(M); npo = np.empty(M)
    pe = np.empty(M); pp = np.empty(M)
    for i in range(M):
        _, _, secs, tag, _ = sample_pair_event(E, _BW4_Z, data)
        assert tag == "pair"
        (_, te, de, _), (_, tp_, dp, _) = secs
        th_e, ph_e = _direction_to_angles(de); th_p, ph_p = _direction_to_angles(dp)
        tot = te + tp_
        f[i] = te / tot if tot > 0 else 0.5
        ne[i] = 1 - math.cos(th_e); npo[i] = 1 - math.cos(th_p)
        pe[i] = ph_e; pp[i] = ph_p
    pve = (ne * (2 - ne)) > BW4_PHI_EPS**2
    pvp = (npo * (2 - npo)) > BW4_PHI_EPS**2
    return {"f": f, "nu_e": ne, "nu_p": npo, "phi_e": pe, "phi_p": pp,
            "phi_e_valid": pve.astype(np.uint8),
            "phi_p_valid": pvp.astype(np.uint8)}


def _factor_conditions(factor, E_grid, thr):
    out = []
    for E in E_grid:
        E = float(E)
        if factor == "photo":
            out += [(E, s) for s in range(N_SHELLS)]
        elif factor == "pair":
            if E > thr:
                out.append((E, None))
        else:
            out.append((E, None))
    return out


def _estimate_null_kl(n, S=BW4_S_SUB):
    if n > 2048:
        return (S - 1) / (2.0 * n)
    if (S, n) not in _BW4_NULLC:
        r = np.random.default_rng(40990 + n)
        d = r.multinomial(int(n), np.full(S, 1 / S), size=2048) / float(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = np.where(d > 0, d * np.log(d * S), 0.0).sum(1)
        _BW4_NULLC[(S, n)] = float(kl.mean())
    return _BW4_NULLC[(S, n)]


def _representation_floor(x, edges):
    a, b = edges[:-1], edges[1:]
    F, n_tot = 0.0, len(x)
    for k in range(len(a)):
        lo, hi = a[k], b[k]
        if hi <= lo:
            continue
        xk = x[(x >= lo) & ((x < hi) if k < len(a) - 1 else (x <= hi))]
        if len(xk) == 0:
            continue
        p = np.histogram(xk, bins=np.linspace(lo, hi, BW4_S_SUB + 1))[0] \
            / len(xk)
        kl = float(np.sum(p[p > 0] * np.log(p[p > 0] * BW4_S_SUB)))
        F += (len(xk) / n_tot) * (kl - _estimate_null_kl(len(xk)))
    return max(0.0, F)


def generate_dataset(data, out="schema_v4_data.npz", M=None,
                           grid_stride=1, verbose=True):
    """Menu option 1.  Generates grouped MC outcomes for every factor on the
    train grid (incl. near-threshold pair points), validation midpoints, and
    the untouched test grid; derives quantile bin edges from TRAIN samples
    (equal weight per condition; φ analytic equal bins); performs ONE
    internal representation sanity check on VALIDATION samples (null-
    calibrated floor, threshold 0.02 nats worst-condition), escalating a
    clearly failing factor's bin count ONCE (x2) before erroring."""
    M = {**BW4_M_DEFAULT, **(M or {})}
    g = build_energy_grids(); thr = g["pair_threshold"]
    grids = {r: (g[r][::grid_stride] if grid_stride > 1 else g[r])
             for r in ("train", "val", "test")}
    if grid_stride > 1:
        grids["train"] = np.unique(np.concatenate(
            [grids["train"], np.array(BW4_PAIR_TRAIN_EXTRA)]))
    arrays, mani = {}, {"tool": f"v{BW_VERSION}", "grid_sha256": g["sha256"],
                        "grid_stride": grid_stride, "M": M, "groups": {},
                        "physics_constants": runtime_physics_metadata()}
    gens = {"process": lambda E, s, m, sd: sample_process_counts(E, m, data, sd),
            "shell": lambda E, s, m, sd: sample_shell_counts(E, m, data, sd),
            "ray": lambda E, s, m, sd: sample_rayleigh_training_data(E, m, data, sd),
            "comp": lambda E, s, m, sd: sample_compton_training_data(E, m, data, sd),
            "photo": lambda E, s, m, sd: sample_photoelectric_training_data(E, s, m, data, sd),
            "pair": lambda E, s, m, sd: sample_pair_training_data(E, m, data, sd)}
    for role in ("train", "val", "test"):
        for factor in ("process", "shell", "ray", "comp", "photo", "pair"):
            for (E, sh) in _factor_conditions(factor, grids[role], thr):
                tag = f"E{E:.6e}" + ("" if sh is None else f"|H{sh}")
                grp = gens[factor](E, sh, M[factor],
                                   deterministic_seed(role, factor, tag))
                for fld, arr in grp.items():
                    arrays[f"{role}|{factor}|{tag}|{fld}"] = np.asarray(arr)
                mani["groups"].setdefault(role, {}).setdefault(
                    factor, []).append(tag)
        if verbose:
            print(f"  [v4:data] role={role}: "
                  f"{sum(len(v) for v in mani['groups'][role].values())} "
                  f"sample groups generated (one photon energy, and one shell for photoelectric sampling)")
    edges = derive_edges(arrays, grids["train"], thr, verbose=verbose)
    edges, sanity = edge_sanity_check(arrays, edges, grids, thr,
                                          verbose=verbose)
    mani["sanity_check"] = sanity
    for nm, e in edges.items():
        arrays[f"__edges__|{nm}"] = e
        mani.setdefault("edges", {})[nm] = {
            "K": int(len(e) - 1),
            "sha256": hashlib.sha256(e.tobytes()).hexdigest()}
    csvs = (PHOTON_CROSS_SECTION_TABLE, RAYLEIGH_CROSS_SECTION_TABLE,
            PHOTO_SHELL_TABLE, INCOHERENT_SCATTERING_TABLE,
            COHERENT_FORM_FACTOR_TABLE, ELECTRON_STOPPING_POWER_TABLE)
    meta = {"schema_version": 4, "tool": f"v{BW_VERSION}",
            "physics_constants": runtime_physics_metadata(),
            "rayleigh_hc_keV_angstrom": float(data.HC_KEV_A),
            "norm_logE": BW4_NORM_LOGE_SPEC, "coordinates": BW4_COORDS,
            "process_order": PROCESS_NAMES, "shell_order": SHELL_NAMES,
            "pair_threshold_MeV": thr, "phi_valid_eps": BW4_PHI_EPS,
            "grids": {r: grids[r].tolist() for r in grids},
            "grid_sha256": g["sha256"], "M": M,
            "seed_derivation": "sha256('v040|role|factor|cond')[:4]",
            "physics_tables": {c: (hashlib.sha256(Path(c).read_bytes())
                                   .hexdigest()
                                   if os.path.isfile(c) else None)
                               for c in csvs}}
    arrays["__bw4_meta__"] = np.frombuffer(
        json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(out, **arrays)
    with open(out, "rb") as fh:
        mani["npz_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.splitext(out)[0] + "_manifest.json", "w") as fh:
        json.dump(mani, fh, indent=2)
    if verbose:
        print(f"  [v4:data] wrote {out} (+manifest); edges: "
              f"{ {k: len(v)-1 for k, v in edges.items()} }")
    return mani


def derive_edges(arrays, E_train, thr, k_over=None, verbose=True):
    k_over = k_over or {}
    rng = np.random.default_rng(40777)
    spec = {"ray_q2": ("ray", "lq2r", None, 0.0),   # lo from data; hi exact
            "comp_u": ("comp", "u", 0.0, 1.0),
            "photo_nu": ("photo", "nu", 0.0, 2.0),
            "pair_frac": ("pair", "f", 0.0, 1.0),
            "pair_nu_e": ("pair", "nu_e", 0.0, 2.0),
            "pair_nu_p": ("pair", "nu_p", 0.0, 2.0)}
    edges = {}
    for nm, (factor, fld, lo, hi) in spec.items():
        K = int(k_over.get(nm, BW4_K.get(nm, BW4_K["pair_nu"])))
        keys = [k for k in arrays
                if k.startswith(f"train|{factor}|") and k.endswith("|" + fld)]
        pools = [arrays[k] for k in keys]
        n_eq = min(len(p) for p in pools)
        pooled = np.concatenate(
            [rng.choice(p, n_eq, replace=False) for p in pools])
        if lo is None:
            lo = float(pooled.min())
        interior = np.quantile(pooled, np.linspace(0, 1, K + 1)[1:-1])
        edges[nm] = np.unique(np.concatenate([[lo], interior, [hi]]))
    edges["phi"] = np.linspace(0.0, 2 * math.pi, BW4_K["phi"] + 1)
    if verbose:
        print(f"  [v4:edges] derived from train samples: "
              f"{ {k: len(v)-1 for k, v in edges.items()} }")
    return edges


def edge_sanity_check(arrays, edges, grids, thr, verbose=True):
    """ONE internal check on validation samples; ONE x2 escalation for a
    factor whose worst null-calibrated floor exceeds BW4_SANITY_FLOOR."""
    def _worst(nm, factor, fld, e):
        """Worst validation floor over conditions with ADEQUATE statistics
        (>= 16 events/bin on average); below that the null-calibrated
        estimator is noise-dominated and cannot gate.  Canonical M
        (8192-32768) always qualifies; labeled SMOKE runs may skip."""
        worst, checked, skipped = 0.0, 0, 0
        for k in [k for k in arrays if k.startswith(f"val|{factor}|")
                  and k.endswith("|" + fld)]:
            occ = int((np.histogram(arrays[k], bins=e)[0] > 0).sum())
            if len(arrays[k]) < 16 * max(occ, 1):
                skipped += 1        # per-condition OCCUPIED-bin statistics
                continue
            worst = max(worst, _representation_floor(arrays[k], e))
            checked += 1
        return worst, checked, skipped
    spec = {"ray_q2": ("ray", "lq2r"), "comp_u": ("comp", "u"),
            "photo_nu": ("photo", "nu"), "pair_frac": ("pair", "f"),
            "pair_nu_e": ("pair", "nu_e"), "pair_nu_p": ("pair", "nu_p")}
    report, k_over = {}, {}
    for nm, (factor, fld) in spec.items():
        w, nc, nsk = _worst(nm, factor, fld, edges[nm])
        report[nm] = {"worst_floor": w if nc else None, "K": int(len(edges[nm]) - 1),
                      "checked_conditions": nc, "skipped_low_stats": nsk,
                      "pass": (w <= BW4_SANITY_FLOOR) if nc else None,
                      "insufficient_statistics": nc == 0}
        if nc == 0 and verbose:
            print(f"  [v4:sanity] {nm}: all {nsk} validation sample groups "
                  f"below 16 events/bin — check SKIPPED (noise-dominated); "
                  f"rerun at canonical M for a meaningful verdict")
        if report[nm]["pass"] is False:
            k_over[nm] = 2 * (len(edges[nm]) - 1)
    if k_over:
        if verbose:
            print(f"  [v4:sanity] escalating once (x2 bins): {k_over}")
        edges = derive_edges(arrays, grids["train"], thr,
                                 k_over=k_over, verbose=verbose)
        for nm in k_over:
            factor, fld = spec[nm]
            w, nc, nsk = _worst(nm, factor, fld, edges[nm])
            report[nm] = {"worst_floor": w if nc else None, "K": int(len(edges[nm]) - 1),
                          "checked_conditions": nc,
                          "skipped_low_stats": nsk,
                          "pass": (w <= BW4_SANITY_FLOOR) if nc else None,
                          "insufficient_statistics": nc == 0,
                          "escalated": True}
    bad = [nm for nm, r in report.items() if r["pass"] is False]
    if verbose:
        for nm, r in report.items():
            floor_text = (f"{r['worst_floor']:.4f}"
                          if r["worst_floor"] is not None else "not measured")
            verdict = ("SKIPPED" if r["insufficient_statistics"]
                       else "ok" if r["pass"] else "FAIL")
            print(f"  [v4:sanity] {nm:10s} K={r['K']:<4d} "
                  f"worst floor={floor_text} -> {verdict}")
    if bad:
        raise RuntimeError(
            f"representation sanity check failed after one escalation for "
            f"{bad} — factor-specific investigation required before "
            f"training (do NOT train these heads)")
    return edges, report


def write_generator_spec(path="schema_v4_generator_spec.json"):
    g = build_energy_grids()
    spec = {"tool": f"v{BW_VERSION}", "norm_logE": BW4_NORM_LOGE_SPEC,
            "physics_constants": runtime_physics_metadata(),
            "coordinates": BW4_COORDS, "bin_counts": BW4_K,
            "M_defaults": BW4_M_DEFAULT,
            "grids": {k: (g[k].tolist() if hasattr(g[k], "tolist") else g[k])
                      for k in ("train", "val", "test", "pair_threshold")},
            "grid_sha256": g["sha256"],
            "pair_train_extras_MeV": BW4_PAIR_TRAIN_EXTRA,
            "process_order": PROCESS_NAMES, "shell_order": SHELL_NAMES,
            "sanity_check": {"metric": "null-calibrated piecewise-uniform "
                             "floor on validation samples",
                             "threshold_nats": BW4_SANITY_FLOOR,
                             "escalation": "single x2 bin-count rebuild"},
            "seed_derivation": "sha256('v040|role|factor|cond')[:4]"}
    with open(path, "w") as fh:
        json.dump(spec, fh, indent=2)
    return spec


def load_dataset_edges(ds_path):
    with np.load(ds_path) as z:
        edges = {k.split("|", 1)[1]: z[k] for k in z.files
                 if k.startswith("__edges__|")}
    assert edges, f"{ds_path} carries no edge arrays — regenerate (option 1)"
    return edges
