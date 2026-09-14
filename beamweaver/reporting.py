"""Rebuild comparison figures from saved Beam Weaver artifacts."""

# Copyright (c) 2025–2026 Pedro Teles. Apache-2.0.

import json
from pathlib import Path


def regenerate_reports(run_dir, save_dir=None, verbose=True):
    """Plot separately normalized depth-dose curves from a comparison run.

    Raw dose arrays and dose sums remain in the comparison run. Separate
    peak normalization shows shape only and does not compare absolute dose.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir).resolve()
    report_file = run_dir / "comparison.json"
    if not report_file.is_file():
        raise FileNotFoundError(f"Comparison report not found: {report_file}")
    rep = json.loads(report_file.read_text())
    if rep.get("schema") != "beamweaver_comparison_v1":
        raise ValueError("Expected a current Beam Weaver comparison report")
    save_dir = Path(save_dir) if save_dir is not None else run_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for etag, row in rep["arms"].items():
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        try:
            for arm in ("MC1", "MC2", "BeamWeaver"):
                result = row.get(arm, {})
                if "skipped" in result:
                    continue
                dose_file = run_dir / result["dose_file"]
                if not dose_file.is_file():
                    raise FileNotFoundError(f"Dose array not found: {dose_file}")
                dose = np.load(dose_file, allow_pickle=False)
                depth = (dose.sum(axis=tuple(range(dose.ndim - 1)))
                         if dose.ndim > 1 else dose)
                ax.plot(depth / max(float(depth.max()), 1e-30), label=arm)
            ax.set(xlabel="depth bin", ylabel="dose / own peak",
                   title=f"MC1 / MC2 / BeamWeaver, E0={etag[1:]} MeV")
            ax.legend()
            fig.tight_layout()
            path = save_dir / f"pdd_{etag}.png"
            fig.savefig(path, dpi=120)
            made.append(str(path))
        finally:
            plt.close(fig)
    if verbose:
        print(f"  [report] rebuilt {len(made)} figure(s) -> {save_dir}")
    return made
