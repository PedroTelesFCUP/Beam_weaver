"""Rebuild comparison figures and numeric tables from saved Beam Weaver runs.

Interaction histograms contain every recorded collision in recursive showers;
the figure energy is the initial primary energy, not every collision energy.
"""

# Copyright (c) 2025–2026 Pedro Teles. Apache-2.0.

import csv
import json
from pathlib import Path

import numpy as np

_ARMS = ("MC1", "MC2", "BeamWeaver")
_PROCESSES = ("rayleigh", "compton", "photo", "pair")
_SHELLS = ("H_K", "O_K", "O_L1", "O_L2", "O_L3")
_COLORS = {"MC1": "#172d4f", "MC2": "#2878b8", "BeamWeaver": "#dd7824"}
_STYLES = {"MC1": "-", "MC2": "--", "BeamWeaver": "-"}


def _available(row):
    """Yield recorded methods, omitting explicitly skipped methods."""
    for arm in _ARMS:
        result = row.get(arm, {})
        if isinstance(result, dict) and "skipped" not in result:
            yield arm, result


def _histogram(result, key, edge_key, *, nested=None):
    """Return bin edges, probabilities and observed count, if any."""
    hist = result.get("event_histograms") or {}
    counts = (hist.get(key) if nested is None else
              (hist.get(nested) or {}).get(key))
    edges = hist.get(edge_key)
    if counts is None or edges is None:
        return None
    counts, edges = np.asarray(counts, float), np.asarray(edges, float)
    if (counts.ndim != 1 or edges.ndim != 1 or len(edges) != len(counts) + 1
            or not np.all(np.isfinite(counts)) or np.any(counts < 0)
            or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0)):
        raise ValueError(f"Invalid saved {key} event histogram")
    n = float(counts.sum())
    return (edges, counts / n, n) if n else None


def _save_figure(fig, plt, save_dir, stem, made):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        path = save_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=180)
        made.append(str(path))
    plt.close(fig)


def _angular_plot(plt, row, etag, energy, process, stem, label, save_dir, made):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    plotted = False
    try:
        for arm, result in _available(row):
            sample = _histogram(result, process, "polar_edges_deg",
                                nested="polar_counts")
            if sample is None:
                continue
            edges, probability, n = sample
            ax.plot((edges[:-1] + edges[1:]) / 2, probability,
                    color=_COLORS[arm], ls=_STYLES[arm], lw=1.9,
                    marker="o", markersize=3,
                    label=f"{arm} (n={n:,.0f})")
            plotted = True
        if plotted:
            ax.set(xlim=(0, 180), xlabel="Polar scattering angle (degrees)",
                   ylabel="Probability per angle bin",
                   title=f"{label} events · primary $E_0$={energy:g} MeV")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
            _save_figure(fig, plt, save_dir, f"{stem}_{etag}", made)
    finally:
        plt.close(fig)


def _photo_plot(plt, row, etag, energy, save_dir, made):
    angles, shells = {}, {}
    for arm, result in _available(row):
        sample = _histogram(result, "photo", "polar_edges_deg",
                            nested="polar_counts")
        if sample is not None:
            angles[arm] = sample
        counts = (result.get("event_histograms") or {}).get("photo_shell_counts")
        if counts is None:
            continue
        if not isinstance(counts, dict):
            raise ValueError("Invalid saved photo_shell_counts")
        values = np.asarray([counts.get(shell, 0) for shell in _SHELLS], float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Invalid saved photo_shell_counts")
        n = values.sum()
        if n:
            shells[arm] = values / n, n
    if not angles and not shells:
        return
    both = bool(angles and shells)
    fig, axes = plt.subplots(1, 2 if both else 1, figsize=(11.2 if both else 6.4, 4.0))
    axes = np.ravel(np.asarray(axes))
    i = 0
    try:
        if angles:
            ax = axes[i]
            i += 1
            for arm, (edges, probability, n) in angles.items():
                ax.plot((edges[:-1] + edges[1:]) / 2, probability,
                        color=_COLORS[arm], ls=_STYLES[arm], lw=1.9,
                        marker="o", markersize=3, label=f"{arm} (n={n:,.0f})")
            ax.set(xlim=(0, 180), xlabel="Photoelectron polar angle (degrees)",
                   ylabel="Probability per angle bin")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, fontsize=8)
        if shells:
            ax = axes[i]
            x = np.arange(len(_SHELLS))
            width = 0.8 / len(shells)
            for j, (arm, (probability, n)) in enumerate(shells.items()):
                ax.bar(x - 0.4 + width * (j + 0.5), probability,
                       width=width, color=_COLORS[arm],
                       label=f"{arm} (n={n:,.0f})")
            ax.set(xticks=x, xticklabels=("H-K", "O-K", "O-L1", "O-L2", "O-L3"),
                   xlabel="Photoelectron shell tag",
                   ylabel="Fraction of recorded shell tags")
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Photoelectric outcomes · primary $E_0$={energy:g} MeV")
        _save_figure(fig, plt, save_dir, f"photoelectric_angle_shell_{etag}", made)
    finally:
        plt.close(fig)


def _pair_plot(plt, row, etag, energy, save_dir, made):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    plotted = False
    try:
        for arm, result in _available(row):
            sample = _histogram(result, "pair_fraction_counts", "pair_fraction_edges")
            if sample is None:
                continue
            edges, probability, n = sample
            undefined = (result.get("event_histograms") or {}).get(
                "pair_fraction_undefined", 0)
            note = f", {int(undefined):,} undefined excluded" if undefined else ""
            ax.plot((edges[:-1] + edges[1:]) / 2, probability,
                    color=_COLORS[arm], ls=_STYLES[arm], lw=1.9,
                    marker="o", markersize=2.4,
                    label=f"{arm} (n={n:,.0f}{note})")
            plotted = True
        if plotted:
            ax.set(xlim=(0, 1),
                   xlabel=r"Electron kinetic fraction $T_-/(T_-+T_+)$",
                   ylabel="Probability per fraction bin",
                   title=f"Pair kinetic-energy sharing · primary $E_0$={energy:g} MeV")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, fontsize=8)
            _save_figure(fig, plt, save_dir, f"pair_share_{etag}", made)
    finally:
        plt.close(fig)


def _summary_csv(report, save_dir):
    """Use the numbers in comparison.json; no recomputation or fitted metrics."""
    fields = (
        "primary_energy_MeV", "method", "primary_histories", "seed",
        "interaction_count", "secondary_count", "deposited_energy_sum_MeV",
        "elapsed_wall_s", "histories_per_s", "rayleigh_interaction_fraction",
        "compton_interaction_fraction", "photo_interaction_fraction",
        "pair_interaction_fraction", "MC1_MC2_energy_sum_absdiff_MeV",
        "MC1_MC2_interaction_fraction_TV", "skipped_reason",
    )
    path = save_dir / "comparison_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for etag, row in report["arms"].items():
            floor = row.get("MC_floor", {})
            for arm in _ARMS:
                result = row.get(arm, {})
                fractions = result.get("interaction_fractions", {})
                values = {
                    "primary_energy_MeV": float(etag[1:]),
                    "method": arm,
                    "primary_histories": report.get("n_histories", ""),
                    "seed": result.get("seed", ""),
                    "interaction_count": result.get("n_interactions", ""),
                    "secondary_count": result.get("n_secondaries", ""),
                    "deposited_energy_sum_MeV": result.get("dose_sum", ""),
                    "elapsed_wall_s": result.get("wall_s", ""),
                    "histories_per_s": result.get("hist_per_s", ""),
                    "MC1_MC2_energy_sum_absdiff_MeV": floor.get("dose_sum_absdiff", ""),
                    "MC1_MC2_interaction_fraction_TV": floor.get("fraction_TV", ""),
                    "skipped_reason": result.get("skipped", ""),
                }
                for process in _PROCESSES:
                    values[f"{process}_interaction_fraction"] = fractions.get(process, "")
                writer.writerow(values)
    return str(path)


def regenerate_reports(run_dir, save_dir=None, verbose=True):
    """Replot saved comparisons in PNG and PDF, return paths of produced files.

    PDD divides each deposited-energy profile by its own peak and therefore
    compares shapes, not absolute absorbed dose (Gy). Old comparisons lacking
    event histograms still provide PDD, process fractions, timing, and CSV.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir).resolve()
    source = run_dir / "comparison.json"
    if not source.is_file():
        raise FileNotFoundError(f"Comparison report not found: {source}")
    report = json.loads(source.read_text())
    if report.get("schema") != "beamweaver_comparison_v1":
        raise ValueError("Expected a current Beam Weaver comparison report")
    save_dir = Path(save_dir) if save_dir is not None else run_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    made = [_summary_csv(report, save_dir)]
    have_histograms = False

    for etag, row in report["arms"].items():
        energy = float(etag[1:])
        have_histograms |= any(bool(result.get("event_histograms"))
                               for _, result in _available(row))

        # The water phantom contains 100 depth slabs, each 1 cm thick.
        # The array stores MeV deposited in each slab, not Gy.
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        plotted = False
        try:
            for arm, result in _available(row):
                if "dose_file" not in result:
                    continue
                dose_file = run_dir / result["dose_file"]
                if not dose_file.is_file():
                    raise FileNotFoundError(f"Dose array not found: {dose_file}")
                dose = np.asarray(np.load(dose_file, allow_pickle=False), float)
                if dose.ndim < 1 or not np.all(np.isfinite(dose)) or np.any(dose < 0):
                    raise ValueError(f"Invalid deposited-energy array: {dose_file}")
                depth = (dose.sum(axis=tuple(range(dose.ndim - 1)))
                         if dose.ndim > 1 else dose)
                if not len(depth) or depth.max() <= 0:
                    continue
                ax.plot(np.arange(len(depth)) + 0.5, depth / depth.max(),
                        color=_COLORS[arm], ls=_STYLES[arm], lw=1.9, label=arm)
                plotted = True
            if plotted:
                ax.set(xlim=(0, 100), xlabel="Depth in water (cm; 1-cm slabs)",
                       ylabel="Deposited energy / own peak",
                       title=f"Depth profile · primary $E_0$={energy:g} MeV")
                ax.grid(alpha=0.2)
                ax.legend(frameon=False)
                _save_figure(fig, plt, save_dir, f"pdd_{etag}", made)
        finally:
            plt.close(fig)

        pair_visible = any(
            result.get("interaction_fractions", {}).get("pair", 0) > 0
            and result.get("n_interactions", 0) > 0
            for _, result in _available(row)
        )
        fig, axes = plt.subplots(1, 2 if pair_visible else 1,
                                 figsize=(10.0 if pair_visible else 6.6, 4.0))
        ax = axes[0] if pair_visible else axes
        plotted = False
        try:
            x = np.arange(len(_PROCESSES))
            width = 0.25
            for arm, result in _available(row):
                fractions = result.get("interaction_fractions")
                if not fractions or result.get("n_interactions", 0) <= 0:
                    continue
                values = np.asarray([fractions.get(p, 0.0) for p in _PROCESSES], float)
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError(f"Invalid interaction fractions for {arm}, {etag}")
                offset = (_ARMS.index(arm) - 1) * width
                ax.bar(x + offset, values, width=width, color=_COLORS[arm],
                       label=f"{arm} (n={result['n_interactions']:,})")
                plotted = True
            if plotted:
                ax.set(xticks=x, xticklabels=("Rayleigh", "Compton", "Photoelectric", "Pair"),
                       ylabel="Fraction of all recorded interactions",
                       title=f"Interaction processes · primary $E_0$={energy:g} MeV")
                ax.legend(frameon=False, fontsize=8)
                if pair_visible:
                    pair_ax = axes[1]
                    for arm, result in _available(row):
                        fractions = result.get("interaction_fractions")
                        if not fractions or result.get("n_interactions", 0) <= 0:
                            continue
                        # Display the same stored fraction on an expanded scale;
                        # this is not a renormalization of the process probabilities.
                        pair_ax.bar(_ARMS.index(arm), 100 * fractions.get("pair", 0),
                                    width=0.65, color=_COLORS[arm])
                    pair_ax.set(xticks=np.arange(len(_ARMS)), xticklabels=_ARMS,
                                ylabel="Pair interactions (%)",
                                title="Pair production (expanded scale)")
                    pair_ax.tick_params(axis="x", rotation=18)
                    pair_ax.grid(axis="y", alpha=0.2)
                _save_figure(fig, plt, save_dir, f"interaction_fractions_{etag}", made)
        finally:
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
        plotted = False
        try:
            x = np.arange(len(_ARMS))
            for ax, field, ylabel in (
                (axes[0], "wall_s", "Elapsed wall time (s)"),
                (axes[1], "hist_per_s", "Primary histories / s"),
            ):
                for i, arm in enumerate(_ARMS):
                    value = row.get(arm, {}).get(field)
                    if value is not None and np.isfinite(value) and value >= 0:
                        ax.bar(x[i], value, width=0.65, color=_COLORS[arm])
                        plotted = True
                ax.set(xticks=x, xticklabels=_ARMS, ylabel=ylabel)
                ax.tick_params(axis="x", rotation=18)
                ax.grid(axis="y", alpha=0.2)
            if plotted:
                fig.suptitle(f"Timing · primary $E_0$={energy:g} MeV")
                fig.supxlabel("Equal primary history counts; wall time includes runner "
                              "setup and photon/electron transport", fontsize=8)
                _save_figure(fig, plt, save_dir, f"timing_{etag}", made)
        finally:
            plt.close(fig)

        _angular_plot(plt, row, etag, energy, "rayleigh", "rayleigh_angle",
                      "Rayleigh", save_dir, made)
        _angular_plot(plt, row, etag, energy, "compton", "compton_angle",
                      "Compton", save_dir, made)
        _photo_plot(plt, row, etag, energy, save_dir, made)
        _pair_plot(plt, row, etag, energy, save_dir, made)

    if verbose:
        if not have_histograms:
            print("  [report] no recorded event histograms; angular, shell and pair "
                  "figures require a comparison with histogram recording")
        print(f"  [report] rebuilt {len(made)} file(s) -> {save_dir}")
    return made
