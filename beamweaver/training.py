"""Beam Weaver: training."""

# Copyright (c) 2025–2026 Pedro Teles & João Melo. Apache-2.0.

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .constants import runtime_physics_metadata
from .constants import PAIR_THRESHOLD_MEV
from .constants import (
    BW4_FACTOR_HEADS, BW4_TRAIN_SEQUENCE,
    DEFAULT_TRAIN_BATCH_SIZE, N_PROCESSES, N_SHELLS, PROCESS_INDEX,
)
from .coordinates import norm_logE
from .dataset import load_dataset_edges
from .policy import GenerativeTransportPolicy, UnconditionalCategoricalHead


def restore_factor(policy, factor, path=None):
    """Load a completed per-factor checkpoint and mark the factor ready."""
    path = path or f"v040_head_{factor}.pt"
    if not os.path.isfile(path):
        return False
    ck = torch.load(path, map_location=policy._dev(), weights_only=False)
    getattr(policy, BW4_FACTOR_HEADS[factor][0]).load_state_dict(
        ck["state_dict"])
    policy.ready[factor] = True
    return True


def _masked_ce(logits, targets, mask=None):
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1)
    logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
    return -(targets * logp).sum(-1)


def _energy_from_group_key(key):
    return float(key.split("|")[2][1:].split("|")[0])


def factor_tensors(ds_path, factor, edges, role):
    """Build training tensors for one factor from the schema-v4 npz.
    Grouped factors -> (X, T, mask); per-event pair-direction factors ->
    (X_events, y_events)."""
    z = np.load(ds_path)
    thr = PAIR_THRESHOLD_MEV
    if factor in ("pair_e_polar", "pair_e_phi", "pair_p_polar",
                  "pair_p_phi"):
        lep = "e" if "_e_" in factor else "p"
        fld = (f"nu_{lep}" if "polar" in factor else f"phi_{lep}")
        e = edges[f"pair_nu_{lep}"] if "polar" in factor else edges["phi"]
        X, y = [], []
        for k in [k for k in z.files if k.startswith(f"{role}|pair|")
                  and k.endswith("|f")]:
            E = _energy_from_group_key(k); base = k[:-2]
            f_arr = z[k]
            flep = f_arr if lep == "e" else 1.0 - f_arr
            v = z[base + "|" + fld]
            ok = np.ones(len(v), bool)
            if "phi" in factor:
                ok = z[base + f"|phi_{lep}_valid"].astype(bool)
            idx = np.clip(np.searchsorted(e, v[ok], side="right") - 1,
                          0, len(e) - 2)
            X.append(np.stack([np.full(ok.sum(), norm_logE(E)),
                               flep[ok]], axis=1))
            y.append(idx)
        return (torch.tensor(np.concatenate(X), dtype=torch.float32),
                torch.tensor(np.concatenate(y), dtype=torch.long))
    rows_X, rows_T, rows_M = [], [], []
    if factor == "process":
        for k in sorted(k for k in z.files
                        if k.startswith(f"{role}|process|")):
            E = _energy_from_group_key(k)
            c = z[k].astype(np.float64); T = c / c.sum()
            rows_X.append([norm_logE(E)])
            rows_T.append(T)
            mask = [True] * N_PROCESSES
            mask[PROCESS_INDEX["pair"]] = E > thr
            rows_M.append(mask)
    elif factor == "shell":
        for k in sorted(k for k in z.files
                        if k.startswith(f"{role}|shell|")):
            c = z[k].astype(np.float64)
            rows_X.append([norm_logE(_energy_from_group_key(k))])
            rows_T.append(c / c.sum())
    elif factor in ("ray_q2", "comp_u", "photo_nu", "pair_frac"):
        fmap = {"ray_q2": ("ray", "lq2r"), "comp_u": ("comp", "u"),
                "photo_nu": ("photo", "nu"), "pair_frac": ("pair", "f")}
        fac, fld = fmap[factor]
        e = edges[factor]
        for k in sorted(k for k in z.files
                        if k.startswith(f"{role}|{fac}|")
                        and k.endswith("|" + fld)):
            E = _energy_from_group_key(k)
            x = z[k]
            h = np.histogram(x, bins=e)[0].astype(np.float64)
            rows_T.append(h / max(h.sum(), 1))
            xrow = [norm_logE(E)]
            if factor == "photo_nu":
                sh = int(k.split("|H")[1].split("|")[0])
                xrow = [norm_logE(E)] + [1.0 * (j == sh)
                                             for j in range(N_SHELLS)]
            rows_X.append(xrow)
    elif factor in ("ray_phi", "comp_phi", "photo_phi"):
        fac = {"ray_phi": "ray", "comp_phi": "comp",
               "photo_phi": "photo"}[factor]
        e = edges["phi"]
        for k in sorted(k for k in z.files
                        if k.startswith(f"{role}|{fac}|")
                        and k.endswith("|phi")):
            v = z[k]; ok = z[k + "_valid"].astype(bool)
            h = np.histogram(v[ok], bins=e)[0].astype(np.float64)
            rows_T.append(h / max(h.sum(), 1))
            if factor == "photo_phi":
                sh = int(k.split("|H")[1].split("|")[0])
                rows_X.append([1.0 * (j == sh) for j in range(N_SHELLS)])
            else:
                rows_X.append([])
    X = (torch.tensor(np.array(rows_X, dtype=np.float32))
         if rows_X and len(rows_X[0]) else None)
    T = torch.tensor(np.array(rows_T, dtype=np.float32))
    M = (torch.tensor(np.array(rows_M, dtype=bool)) if rows_M else None)
    return X, T, M


def train_factor(policy, factor, ds_path, epochs=None, lr=None,
                     patience=None, batch=DEFAULT_TRAIN_BATCH_SIZE,
                     device="cpu", verbose=True,
                     run_dir="."):
    """Train one factor with empirical cross-entropy and validation stopping.

    Pair-direction heads use per-event targets; other heads use grouped
    empirical distributions. Freeze the other heads and save the best
    validation state, its checkpoint, and a manifest entry in ``run_dir``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"v040_head_{factor}.pt"
    manifest_path = run_dir / "v040_factor_manifest.json"
    attr = BW4_FACTOR_HEADS[factor][0]
    head = getattr(policy, attr)
    per_event = factor.startswith("pair_") and factor != "pair_frac"
    epochs = epochs or (20 if per_event else 400)
    lr = lr or (3e-3 if per_event else 5e-3)
    patience = patience or (4 if per_event else 60)
    for p_ in policy.parameters():
        p_.requires_grad_(False)
    for p_ in head.parameters():
        p_.requires_grad_(True)
    head.to(device).train()
    snap = {k: v.detach().clone()
            for k, v in policy.state_dict().items()
            if not k.startswith(f"{attr}.")}
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    tr = factor_tensors(ds_path, factor, policy.edge_arrays, "train")
    va = factor_tensors(ds_path, factor, policy.edge_arrays, "val")
    if verbose:
        outd = (head.logits.shape[0] if isinstance(head, UnconditionalCategoricalHead)
                else head.net[-1].out_features)
        n_tr = tr[1].shape[0]
        print(f"  [v4:train] head={factor} module={attr} "
              f"in={BW4_FACTOR_HEADS[factor][1]} out={outd} "
              f"train_{'events' if per_event else 'conds'}={n_tr} "
              f"val={va[1].shape[0]} opt=Adam lr={lr} epochs<={epochs}")
    best, best_state, best_ep, bad, curve = None, None, -1, 0, []

    def _grouped_eval(X, T, M):
        with torch.no_grad():
            lg = head(X.to(device) if X is not None else T.shape[0])
            ce = _masked_ce(lg, T.to(device),
                                M.to(device) if M is not None else None)
            p = torch.softmax(lg.masked_fill(~M.to(device), float("-inf"))
                              if M is not None else lg, -1)
            tv = 0.5 * (p - T.to(device)).abs().sum(-1)
            return float(ce.mean()), float(tv.mean())

    for ep in range(epochs):
        if per_event:
            X, y = tr
            perm = torch.randperm(X.shape[0],
                                  generator=torch.Generator().manual_seed(ep))
            tot = 0.0
            for i in range(0, X.shape[0], batch):
                sl = perm[i:i + batch]
                loss = torch.nn.functional.cross_entropy(
                    head(X[sl].to(device)), y[sl].to(device))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(sl)
            tr_ce = tot / X.shape[0]
            with torch.no_grad():
                Xv, yv = va
                v_ce = float(torch.nn.functional.cross_entropy(
                    head(Xv.to(device)), yv.to(device)))
                v_tv = float("nan")
        else:
            X, T, M = tr
            lg = head(X.to(device) if X is not None else T.shape[0])
            loss = _masked_ce(lg, T.to(device),
                                  M.to(device) if M is not None
                                  else None).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tr_ce = float(loss)
            v_ce, v_tv = _grouped_eval(*va)
        curve.append((ep, tr_ce, v_ce))
        if best is None or v_ce < best - 1e-6:
            best, best_ep, bad = v_ce, ep, 0
            best_state = {k: v.detach().clone()
                          for k, v in head.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % max(1, epochs // 10) == 0 or bad > patience):
            print(f"    ep {ep:4d}  train CE {tr_ce:.5f}  val CE {v_ce:.5f}"
                  + ("" if per_event else f"  val TV {v_tv:.4f}")
                  + f"  best@{best_ep}")
        if bad > patience:
            break
    head.load_state_dict(best_state)
    head.eval()
    for k, v in policy.state_dict().items():
        if not k.startswith(f"{attr}."):
            assert torch.equal(v, snap[k]), \
                f"head isolation violated: {k} changed while training " \
                f"{factor}"
    policy.ready[factor] = True
    if per_event:
        fin_tv = None
    else:
        _, fin_tv = _grouped_eval(*va)
    ck = {"factor": factor, "head_attr": attr,
          "state_dict": head.state_dict(), "best_val_ce": best,
          "best_epoch": best_ep, "epochs_run": len(curve),
          "val_tv": fin_tv, "curve": curve[-50:],
          "dataset": ds_path,
          "training_physics_constants": runtime_physics_metadata()}
    torch.save(ck, checkpoint_path)
    _update_training_manifest(factor, {"status": "trained", "val_ce": best,
                                  "val_tv": fin_tv, "best_epoch": best_ep,
                                  "checkpoint": str(checkpoint_path)},
                         path=manifest_path)
    if verbose:
        print(f"  [v4:train] {factor}: best val CE {best:.5f} "
              f"(epoch {best_ep}) -> {checkpoint_path}  "
              f"[isolation verified]")
    return best


def _update_training_manifest(factor, rec,
                         path="v040_factor_manifest.json"):
    m = {}
    if os.path.isfile(path):
        with open(path) as fh:
            m = json.load(fh)
    m[factor] = rec
    with open(path, "w") as fh:
        json.dump(m, fh, indent=2)


def train_policy(policy, ds_path, run_dir, resume=False, device="cpu",
                       **kw):
    """Train thirteen disjoint heads in ten stages and save the policy.

    Checkpoints are restored only when ``resume=True``. Per-factor restore
    retains the current checkpoint-loading behavior; callers must select the
    matching run directory and dataset.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    total = len(BW4_TRAIN_SEQUENCE)
    for i, group in enumerate(BW4_TRAIN_SEQUENCE, 1):
        label = group[0] if len(group) == 1 else "pair direction heads"
        if resume:
            for f in group:
                path = run_dir / f"v040_head_{f}.pt"
                if not policy.ready.get(f) and restore_factor(policy, f, path):
                    print(f"  Stage {i}/{total} — {label}: restored "
                          f"{f} from checkpoint (resume)")
        if all(policy.ready.get(f) for f in group) and resume:
            print(f"  Stage {i}/{total} — {label}: already trained (skip)")
            continue
        print(f"  Stage {i}/{total} — {label}")
        for f in group:
            if resume and policy.ready.get(f):
                continue
            train_factor(policy, f, ds_path, device=device,
                             run_dir=run_dir, **kw)
    policy.meta["dataset"] = ds_path
    policy_path = run_dir / "v040_policy.pt"
    ck = policy.save_policy(policy_path)
    print(f"  [v4] complete policy saved -> {policy_path} "
          f"(ready: {sum(policy.ready.values())}/{len(policy.ready)})")
    return ck


def create_policy(ds_path):
    """Initialize the policy using dataset bin edges and record its identity."""
    pol = GenerativeTransportPolicy(load_dataset_edges(ds_path))
    pol.meta["dataset"] = ds_path
    with open(ds_path, "rb") as fh:
        pol.meta["dataset_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    return pol
