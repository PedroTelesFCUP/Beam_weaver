#!/usr/bin/env python3
"""Reproduce the ECMP campaign figures from the committed numeric data.

Default: python scripts/plot_results.py
To rebuild numeric data from the original downloaded campaign folders as well:
    python scripts/plot_results.py --source /path/to/campaign_source

The optional extraction uses a restricted NumPy-only pickle reader. Original
secondary records are never required to render the committed results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import re
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ENERGIES = [("0p1MeV", 0.1), ("1MeV", 1.0), ("2MeV", 2.0),
            ("5MeV", 5.0), ("10MeV", 10.0)]
METHODS = ["MC-1", "MC-2", "Beam Weaver"]
FILES = {"MC-1": "mc_1", "MC-2": "mc_2", "Beam Weaver": "agent"}
COLORS = {"MC-1": "#173d64", "MC-2": "#66a9d8", "Beam Weaver": "#e87722"}
SHELLS = ["H-K", "O-K", "O-L1", "O-L2", "O-L3"]


class NumericUnpickler(pickle.Unpickler):
    """Allow only the NumPy constructors used by archived numeric arrays."""

    def find_class(self, module, name):
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"}:
            if name == "_reconstruct":
                # Pickles from both NumPy 1 and 2 use this numeric constructor.
                core = np._core if hasattr(np, "_core") else np.core
                return core.multiarray._reconstruct
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        raise pickle.UnpicklingError(f"Unsupported global {module}.{name}")


def number(value):
    return float(value.replace(",", "").strip())


def table(text, heading):
    block = text.split(heading, 1)[1].split("\n\n", 1)[0]
    rows = [[cell.strip() for cell in line.split("|")[1:-1]]
            for line in block.splitlines() if line.startswith("|")]
    return {row[0]: [number(value) for value in row[1:]] for row in rows[1:]}


def angles(text, process):
    match = re.search(
        rf"Recursive shower theta \| source E=[^\n]+ \| process={process}\n"
        r"n\(MC-1\)=([\d,]+)\s+n\(Agent\)=([\d,]+)\s+KS=([\d.eE+-]+)[^\n]*\n"
        r"(.*?)(?=\n\n|\Z)", text, re.S)
    if match is None:
        raise ValueError(f"Missing {process} angle summary")
    bounds, mc, bw = [], [], []
    for line in match[4].splitlines():
        cells = line.split("|")
        if len(cells) != 3 or not re.match(r"\s*\d", cells[0]):
            continue
        lo, hi = [float(x) for x in re.findall(r"\d+\.\d+", cells[0])]
        bounds.append([lo, hi])
        mc.append(float(cells[1].split()[-1]))
        bw.append(float(cells[2].split()[-1]))
    if len(bounds) != 18:
        raise ValueError(f"Expected eighteen {process} polar-angle bins")
    return {"bin_bounds_degrees": bounds,
            "probability": {"MC-1": mc, "Beam Weaver": bw},
            "events": {"MC-1": int(match[1].replace(",", "")),
                       "Beam Weaver": int(match[2].replace(",", ""))},
            "source_reported_ks_distance": float(match[3]),
            "unavailable_series": ["MC-2"],
            "precision_note": "Probabilities and KS distance retain the rounded ASCII values; no renormalization."}


def secondary_summary(path, expected_pairs):
    with path.open("rb") as stream:
        records = NumericUnpickler(stream).load()
    if not isinstance(records, list):
        raise ValueError("Expected a list of secondary tuples")
    shell_counts = Counter()
    shares = []
    pair_positrons = 0
    for i, record in enumerate(records):
        if not isinstance(record, tuple) or len(record) not in (4, 5):
            raise ValueError("Unexpected secondary record")
        species, kinetic, direction, tag = record[:4]
        if tag.startswith("photo_"):
            shell_counts[tag.removeprefix("photo_").replace("_", "-")] += 1
        if tag == "pair_p":
            pair_positrons += 1
        if tag != "pair_e":
            continue
        if species != "electron" or i + 1 == len(records):
            raise ValueError("Unpaired electron")
        partner = records[i + 1]
        if partner[0] != "positron" or partner[3] != "pair_p":
            raise ValueError("Pair electron is not followed by its positron")
        total = float(kinetic + partner[1])
        if not np.isfinite(total) or total <= 0 or kinetic < 0 or partner[1] < 0:
            raise ValueError("Invalid pair kinetic energy")
        shares.append(float(kinetic / total))
    if len(shares) != expected_pairs or pair_positrons != expected_pairs:
        raise ValueError("Secondary pair count disagrees with ASCII ledger")
    if set(shell_counts) - set(SHELLS):
        raise ValueError("Unexpected photoelectric shell")
    counts, edges = np.histogram(shares, bins=np.linspace(0, 1, 31))
    return {"shell_counts": [shell_counts[shell] for shell in SHELLS],
            "pair_counts": counts.tolist(), "pair_bin_edges": edges.tolist(),
            "pair_events": len(shares)}


def extract(source, data_path):
    campaign = {"schema": 1,
        "description": "Saved ECMP comparison campaign; not a rerun of the current repository revision.",
        "source_folder": "BeamWeaver_latest_0_2_9",
        "provenance_note": "The poster notes call the folder name stale and describe the results as the current supervised approach. Available original metadata explicitly records version 0.2.9e and checkpoint pretrain_v029e/best_validation_nll.pt. These original identifiers are preserved here.",
        "geometry_from_poster": {"source_square_cm": [10, 10], "water_phantom_cm": [100, 100, 100],
                                 "histories_per_method_per_energy": 50000},
        "methods": METHODS,
        "conventions": {"dose": "Accumulated deposited energy in MeV in each 1 cm longitudinal bin. Displayed PDD divides each method by its own maximum; it is not dose in Gy.",
                        "angles": "Recursive-shower local polar angles, grouped over every collision energy in a run; source energy labels describe the primary photon energy. Eighteen 10-degree bins. Raw rounded ASCII probabilities are preserved in this JSON and normalized to sum one for display, matching the original poster. MC-2 angular summaries were not saved.",
                        "shells": "Counts of photoelectron secondary tags, divided by all tagged photoelectrons of that method.",
                        "pair_share": "Thirty equal bins of electron kinetic share T_minus/(T_minus+T_plus); adjacent pair_e and pair_p records are verified and paired.",
                        "interaction_fractions": "Rounded recorded event fractions for the entire recursive shower; they are not cross-section fractions evaluated only at the primary energy.",
                        "uncertainty": "No error bars; MC-1 and MC-2 are independent reference seeds. Their scatter is not a confidence interval.",
                        "timing": "Recorded wall-clock seconds from the saved campaign; no hardware-controlled benchmark of the current code is claimed."},
        "energies": [], "sources": []}
    for folder, energy in ENERGIES:
        directory = source / folder
        text = (directory / "ascii_summary.txt").read_text()
        runtime = table(text, "Runtime summary")
        process = table(text, "Interaction fractions")
        ledger = table(text, "Secondary-photon and ledger summary")
        comparisons = table(text, "Comparison metrics")
        record = {"id": folder, "energy_MeV": energy,
            "runtime_seconds": {}, "interaction_counts": {}, "pair_events": {},
            "max_ledger_residual_MeV": {}, "interaction_fractions": {}, "dose_MeV": {},
            "dose_bin_centres_cm": (np.arange(100) + 0.5).tolist(),
            "angles": {p: angles(text, p) for p in ("compton", "photo", "rayleigh")},
            "shells": {"labels": SHELLS, "counts": {}, "unavailable_series": []},
            "pair_share": {"bin_edges": np.linspace(0, 1, 31).tolist(), "counts": {}, "unavailable_series": []},
            "source_reported_comparisons": {"column_names": ["dose_ratio", "sum_normalized_pdd_mean_absolute_difference", "sum_normalized_pdd_rms_difference", "pdd_pearson_correlation"], "rows": comparisons}}
        for method in METHODS:
            raw_name = "Agent" if method == "Beam Weaver" else method
            record["runtime_seconds"][method] = runtime[raw_name][0]
            record["interaction_counts"][method] = int(runtime[raw_name][2])
            record["pair_events"][method] = int(ledger[raw_name][0])
            record["max_ledger_residual_MeV"][method] = ledger[raw_name][3]
            record["interaction_fractions"][method] = {p: process[p][METHODS.index(method)] for p in process}
            dose = np.load(directory / f"{FILES[method]}_dose.npy", allow_pickle=False)
            if dose.shape != (100,) or not np.all(np.isfinite(dose)) or np.any(dose < 0):
                raise ValueError("Invalid dose array")
            record["dose_MeV"][method] = dose.tolist()
            path = directory / f"{FILES[method]}_secondaries.pkl.bin"
            if not path.exists():
                path = directory / f"{FILES[method]}_secondaries.pkl"
            if path.exists():
                summary = secondary_summary(path, record["pair_events"][method])
                record["shells"]["counts"][method] = summary["shell_counts"]
                record["pair_share"]["counts"][method] = summary["pair_counts"]
            else:
                record["shells"]["unavailable_series"].append(method)
                record["pair_share"]["unavailable_series"].append(method)
        below = energy < 2 * 0.51099895069
        if below and any(record["pair_events"].values()):
            raise ValueError("Pair events below threshold")
        record["pair_share"]["status"] = "below_threshold_zero_events" if below else ("available" if record["pair_share"]["counts"] else "missing_secondary_energies")
        record["shells"]["status"] = "available" if record["shells"]["counts"] else "missing_secondary_shell_tags"
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            record["original_metadata"] = {key: meta.get(key) for key in ("version", "energy_MeV", "n_photons", "checkpoint", "seeds", "charged_particle_model")}
            record["original_audit"] = {key: value for key, value in meta["audit"].items() if key != "process_energy_counts"}
        else:
            record["original_metadata"] = None
        for path in sorted(directory.iterdir()):
            if path.name.endswith((".npy", ".json", ".txt", ".pkl", ".pkl.bin")):
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(block)
                campaign["sources"].append({"energy_id": folder, "filename": path.name, "bytes": path.stat().st_size, "sha256": h.hexdigest()})
        campaign["energies"].append(record)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(campaign, ensure_ascii=False, separators=(",", ":")) + "\n")
    return campaign


def plot(campaign, output, regenerate_poster=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
        "text.color": "#263c50", "axes.labelcolor": "#263c50",
        "xtick.color": "#263c50", "ytick.color": "#263c50",
        "axes.edgecolor": "#8da5b2", "axes.titlecolor": "#173d64",
        "axes.titleweight": "bold", "axes.titlesize": 15,
        "axes.spines.right": False, "axes.spines.top": False,
        "grid.color": "#dceaee", "grid.linewidth": 0.8,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "svg.hashsalt": "beamweaver-ecmp-results"})

    def setup(ax, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.set_axisbelow(True)

    def curve(ax, x, y, method):
        ax.plot(x, y, label=method, color=COLORS[method],
                linestyle="--" if method == "MC-2" else "-", linewidth=2.6)

    def save(fig, folder, name):
        fig.savefig(folder / f"{name}.png", dpi=240, bbox_inches="tight",
                    metadata={"Description": campaign["description"]})
        plt.close(fig)

    def annotation(ax, text):
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#263c50", linespacing=1.7)

    for record in campaign["energies"]:
        folder = output / record["id"]
        folder.mkdir(parents=True, exist_ok=True)
        if record["id"] == "5MeV" and not regenerate_poster:
            manifest_path = ROOT / "results/data/poster_figures.json"
            if not manifest_path.exists():
                raise FileNotFoundError("Original poster figure manifest is required; use --regenerate-poster with a separate --output directory to redraw.")
            manifest = json.loads(manifest_path.read_text())
            for asset in manifest["assets"]:
                original = ROOT / asset["repository_path"]
                if hashlib.sha256(original.read_bytes()).hexdigest() != asset["sha256"]:
                    raise ValueError(f"Original poster asset changed: {original}")
                destination = folder / original.name
                if original.resolve() != destination.resolve():
                    shutil.copyfile(original, destination)
            continue
        energy = f'{record["energy_MeV"]:g} MeV'
        fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
        for method in METHODS:
            dose = np.asarray(record["dose_MeV"][method])
            curve(ax, record["dose_bin_centres_cm"], dose / dose.max(), method)
        setup(ax, "Depth in water (cm)", "Dose / own peak")
        ax.set(xlim=(0, 100), ylim=(0, 1.04))
        ax.set_title(f"{energy} peak-normalized PDD", loc="left")
        ax.legend(frameon=False)
        save(fig, folder, "pdd")

        def angle_panel(ax, process):
            data = record["angles"][process]
            centres = np.mean(data["bin_bounds_degrees"], axis=1)
            for method, values in data["probability"].items():
                probabilities = np.asarray(values, dtype=float)
                curve(ax, centres, probabilities / probabilities.sum(), method)
            setup(ax, r"Local polar angle, $\theta$ (degrees)" if process == "compton" else r"Local photoelectron angle, $\theta$ (degrees)", "Probability per 10° bin")
            ax.set_xlim(0, 180)
            ax.set_xticks(np.arange(0, 181, 20))
            ax.legend(frameon=False)

        fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
        angle_panel(ax, "compton")
        ax.set_title(f"{energy} Compton scattering angle", loc="left")
        ax.text(1, 1.015, f'KS D = {record["angles"]["compton"]["source_reported_ks_distance"]:.4f}',
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
        save(fig, folder, "compton_angle")

        fig, (ax, shell_ax) = plt.subplots(1, 2, figsize=(10.4, 4.8), layout="constrained")
        fig.suptitle(f"{energy} photoelectric angle and shell selection", x=0.09, ha="left", color="#173d64", weight="bold", fontsize=15)
        angle_panel(ax, "photo")
        ax.text(0.02, 0.98, f'KS D = {record["angles"]["photo"]["source_reported_ks_distance"]:.4f}', transform=ax.transAxes, ha="left", va="top", fontsize=10)
        counts = record["shells"]["counts"]
        if counts:
            y = np.arange(len(SHELLS))
            width = 0.21
            available = [method for method in METHODS if method in counts]
            for i, method in enumerate(available):
                values = np.asarray(counts[method])
                shell_ax.barh(y + (i - (len(available) - 1) / 2) * width, 100 * values / values.sum(), height=width,
                              color=COLORS[method], label=method)
            setup(shell_ax, "Share of photoelectrons (%)", "")
            shell_ax.set_yticks(y, SHELLS)
            shell_ax.set_xscale("log")
            positive = [100 * value / sum(values) for values in counts.values() for value in values if value > 0]
            shell_ax.set_xlim(min(positive) * 0.5, 125)
            for i, method in enumerate(available):
                for j, count in enumerate(counts[method]):
                    if count == 0:
                        shell_ax.text(min(positive) * 0.57, j + (i - (len(available) - 1) / 2) * width,
                                      "0", va="center", fontsize=8, color=COLORS[method])
            shell_ax.legend(frameon=False, loc="lower right", fontsize=10)
            if record["shells"]["unavailable_series"]:
                shell_ax.set_title("MC-2 shell data unavailable", fontsize=10, weight="normal")
        else:
            shell_ax.set_axis_off()
            annotation(shell_ax, "Shell-selection data unavailable\nSecondary shell tags were not saved\nin the available campaign files.")
        save(fig, folder, "photoelectric_angle_shell")

        fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
        ax.set_title(f"{energy} pair-production kinetic-energy sharing", loc="left")
        status = record["pair_share"]["status"]
        if status == "available":
            edges = np.asarray(record["pair_share"]["bin_edges"])
            for method, counts in record["pair_share"]["counts"].items():
                values = np.asarray(counts)
                curve(ax, (edges[1:] + edges[:-1]) / 2, values / values.sum(), method)
            setup(ax, r"Electron share, $f = T_-/(T_- + T_+)$", "Probability per bin")
            ax.set_xlim(0, 1)
            ax.legend(frameon=False, ncol=3, fontsize=10, loc="upper center")
        else:
            ax.set_axis_off()
            message = "Pair production is below threshold\nAll three runs recorded zero pair events.\nA sharing distribution is therefore undefined." if status == "below_threshold_zero_events" else "Pair-sharing data unavailable\nPair events were recorded, but secondary energies\nare absent from the available campaign files."
            annotation(ax, message)
        save(fig, folder, "pair_share")

        fig, (ax, pair_ax) = plt.subplots(1, 2, figsize=(10.4, 4.8),
                                        gridspec_kw={"width_ratios": [2.6, 1]}, layout="constrained")
        fig.suptitle(f"{energy} interaction fractions", x=0.08, ha="left", color="#173d64", weight="bold", fontsize=15)
        x, width = np.arange(3), 0.23
        for i, method in enumerate(METHODS):
            values = record["interaction_fractions"][method]
            ax.bar(x + (i - 1) * width, [100 * values[p] for p in ("rayleigh", "compton", "photo")],
                   width=width, color=COLORS[method], label=method)
            value = 100 * values["pair"]
            pair_ax.bar(i, value, color=COLORS[method], width=0.68)
            pair_ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
        setup(ax, "", "Share of interactions (%)")
        ax.set_xticks(x, ["Rayleigh", "Compton", "Photoelectric"])
        ax.legend(frameon=False, ncol=3, fontsize=10)
        ax.margins(y=0.13)
        setup(pair_ax, "", "Pair production (%)")
        pair_ax.set_xticks(range(3), ["MC-1", "MC-2", "Beam\nWeaver"])
        pair_max = max(record["interaction_fractions"][m]["pair"] * 100 for m in METHODS)
        pair_ax.set_ylim(0, pair_max * 1.2 if pair_max else 1)
        if not pair_max:
            pair_ax.text(0.5, 0.5, "Below threshold\nZero events", transform=pair_ax.transAxes, ha="center", va="center", fontsize=10)
        save(fig, folder, "interaction_fractions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional original campaign folders to extract")
    parser.add_argument("--data", type=Path, default=ROOT / "results/data/campaign.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures")
    parser.add_argument("--regenerate-poster", action="store_true",
                        help="Redraw 5 MeV from numeric data instead of copying originals; requires a separate output directory")
    args = parser.parse_args()
    if args.regenerate_poster and args.output.resolve() == (ROOT / "results/figures").resolve():
        parser.error("--regenerate-poster requires --output outside results/figures to preserve the exact poster originals")
    campaign = extract(args.source, args.data) if args.source else json.loads(args.data.read_text())
    plot(campaign, args.output, args.regenerate_poster)
    print(f"Wrote five result panels for each of {len(campaign['energies'])} energies to {args.output}")


if __name__ == "__main__":
    main()
