"""Audited Beam Weaver showers and independent MC reference comparisons."""

# Copyright (c) 2025–2026 Pedro Teles. Apache-2.0.

from .constants import runtime_physics_metadata
from .constants import PAIR_THRESHOLD_MEV
from .constants import (
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_TRANSPORT_ECUT_MEV,
    PERFORMANCE_TIME_FLOOR_S,
    CURRENT_POLICY_SCHEMA,
)

import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .audit import TransportExecutionLogger, reference_sampler_guard
from .constants import PROCESS_NAMES
from .dataset import deterministic_seed
from .interaction_summaries import summarize_interactions
from .transport import (run_reference_transport, run_learned_transport,
                        run_learned_transport_batched)


def _seed_run(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def compare_transport(policy, data, env, energies, n_hist,
                              device="cpu", batch=DEFAULT_INFERENCE_BATCH_SIZE,
                              save_dir="comparison", verbose=True):
    """Compare MC1, MC2 and BeamWeaver; save raw dose arrays and a summary.

    MC1 and MC2 use independent random streams. Their disagreement gives a
    sampling-noise reference, not a confidence interval or an accuracy bound.
    Timing includes each runner's setup and electron transport.
    """
    energies = [float(energy) for energy in energies]
    if not energies or any(not np.isfinite(e) or e <= 0 for e in energies):
        raise ValueError("energies must contain finite positive values in MeV")
    if n_hist <= 0 or batch <= 0:
        raise ValueError("n_hist and batch must be positive")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rep = {"schema": "beamweaver_comparison_v1", "n_histories": n_hist,
           "runtime_physics_constants": runtime_physics_metadata(),
           "rayleigh_hc_keV_angstrom": float(data.HC_KEV_A),
           "energies": energies, "arms": {},
           "timing_scope": "runner setup, photon and electron transport",
           "event_histograms_metadata": {
               "population": "all recorded photon collisions in recursive showers, including secondary photons",
               "primary_energy_label": "E in each comparison key is the initial primary photon energy in MeV; individual collisions occur at varying incident energies",
               "polar_angle_deg": "Rayleigh and Compton scattered photon; photoelectric photoelectron, relative to incident photon direction",
               "pair_fraction": "electron kinetic energy / (electron + positron kinetic energy) from each recorded pair event",
           }}
    for E0 in energies:
        row = {}
        for arm in ("MC1", "MC2", "BeamWeaver"):
            if arm == "BeamWeaver" and E0 >= PAIR_THRESHOLD_MEV and \
                    not policy.pair_heads_ready:
                row[arm] = {"skipped": "pair heads not ready"}
                continue
            # Keep the established streams while replacing display labels.
            seed = deterministic_seed("cmp", "v040" if arm == "BeamWeaver" else arm,
                             f"{E0}")
            _seed_run(seed)
            env.fixed_energy = E0
            env.dose_tally = np.zeros_like(np.asarray(env.dose_tally, float))
            t0 = time.perf_counter()
            if arm in ("MC1", "MC2"):
                _, secs, dose, inter = run_reference_transport(
                    n_hist, data, env, ecut=DEFAULT_TRANSPORT_ECUT_MEV)
            else:
                generator = torch.Generator(device=device).manual_seed(seed)
                with reference_sampler_guard(None):
                    _, secs, dose, inter, _ = run_learned_transport_batched(
                        policy, data, env, n_hist, E0=E0, ecut=DEFAULT_TRANSPORT_ECUT_MEV,
                        device=device, inference_batch_size=batch,
                        generator=generator)
            wall = time.perf_counter() - t0
            names = [str(i.get("interaction", "?")).split("_")[0]
                     for i in inter if isinstance(i, dict)]
            frac = {p: names.count(p) / max(len(names), 1)
                    for p in PROCESS_NAMES}
            dose_file = f"dose_{arm}_E{E0}.npy"
            np.save(save_dir / dose_file, np.asarray(dose))
            row[arm] = {"seed": seed, "interaction_fractions": frac,
                        "n_interactions": len(names),
                        "event_histograms": summarize_interactions(inter),
                        "dose_sum": float(np.asarray(dose).sum()),
                        "n_secondaries": len(secs), "wall_s": wall,
                        "hist_per_s": n_hist / max(wall, PERFORMANCE_TIME_FLOOR_S),
                        "dose_file": dose_file}
        row["MC_floor"] = {
            "dose_sum_absdiff": abs(row["MC1"]["dose_sum"]
                                    - row["MC2"]["dose_sum"]),
            "fraction_TV": 0.5 * sum(
                abs(row["MC1"]["interaction_fractions"][p]
                    - row["MC2"]["interaction_fractions"][p])
                for p in PROCESS_NAMES)}
        rep["arms"][f"E{E0}"] = row
        if verbose:
            details = "  ".join(
                f"{arm} dose={row[arm]['dose_sum']:.4f} "
                f"({row[arm]['hist_per_s']:.0f} h/s)"
                for arm in ("MC1", "MC2", "BeamWeaver")
                if "wall_s" in row[arm])
            print(f"  [compare] E={E0} MeV: {details}")
    out_json = save_dir / "comparison.json"
    out_json.write_text(json.dumps(rep, indent=2))
    if verbose:
        print(f"  [compare] report and dose arrays -> {save_dir}")
    return rep


def evaluate_transport(policy, data, env, E0, n_hist, device="cpu",
                           save_dir=None, verbose=True):
    """Run serial transport with per-event accounting and sampler tripwires.

    The execution certificate verifies event provenance and internal checks;
    independent physical validation still requires reference comparisons.
    """
    if not np.isfinite(E0) or E0 <= 0 or n_hist <= 0:
        raise ValueError("E0 must be finite and positive; n_hist must be positive")
    save_dir = Path(save_dir or f"evaluation_E{E0}")
    save_dir.mkdir(parents=True, exist_ok=True)
    exec_logger = TransportExecutionLogger(save_dir / "execution")
    exec_logger.checkpoint_schema = CURRENT_POLICY_SCHEMA
    exec_logger.kernel_architecture = type(policy).__name__
    exec_logger.free_path_source = "analytic_tabulated"
    exec_logger.shell_mode = "learned"
    seed = deterministic_seed("eval", "v040", f"{E0}|{n_hist}")
    _seed_run(seed)
    env.fixed_energy = E0
    env.dose_tally = np.zeros_like(np.asarray(env.dose_tally, float))
    generator = torch.Generator(device=device).manual_seed(seed)
    t0 = time.perf_counter()
    with reference_sampler_guard(exec_logger.audit):
        tracks, secs, dose, inter = run_learned_transport(
            policy, data, env, n_hist, E0=E0, ecut=DEFAULT_TRANSPORT_ECUT_MEV,
            device=device, exec_logger=exec_logger, generator=generator)
    wall = time.perf_counter() - t0
    certificate = exec_logger.finalize()
    np.save(save_dir / "dose.npy", np.asarray(dose))
    report = {"schema": "beamweaver_evaluation_v1", "E0": E0,
              "runtime_physics_constants": runtime_physics_metadata(),
              "rayleigh_hc_keV_angstrom": float(data.HC_KEV_A),
              "n_histories": n_hist, "seed": seed, "wall_s": wall,
              "hist_per_s": n_hist / max(wall, PERFORMANCE_TIME_FLOOR_S),
              "n_interactions": len(inter), "n_secondaries": len(secs),
              "dose_sum": float(np.asarray(dose).sum()),
              "certificate": certificate}
    (save_dir / "summary.json").write_text(
        json.dumps(report, indent=2, default=float))
    _plot_tracks(tracks, E0, save_dir, verbose)
    if verbose:
        print(f"  [run] E0={E0} MeV, {n_hist} histories, {len(inter)} events, "
              f"{wall:.1f} s; certificate verified="
              f"{certificate['inference_source_verified']}; "
              f"artifacts -> {save_dir}")
    return certificate


def _plot_tracks(tracks, E0, save_dir, verbose):
    """An optional display failure must not discard completed transport data."""
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        for track in tracks[:40]:
            coords = np.asarray(track["coords"], float)
            if coords.ndim == 2 and coords.shape[0] == 3:
                coords = coords.T  # transport emits coordinate-major (3, N)
            if coords.ndim == 2 and coords.shape[1] == 3 and len(coords):
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], lw=0.6)
        ax.set(xlabel="x (cm)", ylabel="y (cm)", zlabel="z (cm)",
               title=f"Beam Weaver, E0={E0} MeV ({min(len(tracks), 40)} tracks)")
        fig.savefig(save_dir / "shower3d.png", dpi=110)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        if verbose:
            print(f"  [run] optional shower figure unavailable: {exc}")
    finally:
        if fig is not None:
            plt.close(fig)
