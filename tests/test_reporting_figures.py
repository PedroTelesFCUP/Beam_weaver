"""Check that report figures represent recorded interactions, not invented data."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from beamweaver.interaction_summaries import summarize_interactions
from beamweaver.reporting import regenerate_reports


def _result(directory, name, energy, *, histograms=None):
    """A comparison result with a small, nonzero deposited-energy tally."""
    filename = f"dose_{name}_E{energy}.npy"
    np.save(directory / filename, np.array([[0.0, 1.0, 2.0],
                                             [0.0, 2.0, 1.0]]))
    result = {
        "dose_file": filename,
        "dose_sum": 6.0,
        "n_interactions": 10,
        "n_secondaries": 3,
        "wall_s": 2.0 if name == "MC1" else 2.5,
        "hist_per_s": 5.0 if name == "MC1" else 4.0,
        "interaction_fractions": {
            "rayleigh": 0.1, "compton": 0.6, "photo": 0.2, "pair": 0.1,
        },
    }
    if histograms is not None:
        result["event_histograms"] = histograms
    return result


def _write_comparison(directory, arms):
    (directory / "comparison.json").write_text(json.dumps({
        "schema": "beamweaver_comparison_v1",
        "n_histories": 10,
        "energies": [float(tag[1:]) for tag in arms],
        "arms": arms,
    }))


class InteractionSummaryTests(unittest.TestCase):
    def test_angles_shells_and_pair_fractions_come_from_same_records(self):
        records = [
            {"interaction": "rayleigh", "angle": 0.0},
            {"interaction": "rayleigh", "angle": 180.0},
            {"interaction": "compton", "angle": 10.0},
            {"interaction": "photo", "angle": 45.0,
             "secondaries": [("electron", 0.8, None, "photo_O_K")]},
            {"interaction": "photo", "angle": 181.0,
             "secondaries": [("electron", 0.8, None, "other_tag")]},
            {"interaction": "pair", "secondaries": [
                ("electron", 3.0, None, "pair_e"),
                ("positron", 1.0, None, "pair_p")]},
            {"interaction": "pair", "secondaries": [
                ("electron", 0.0, None, "pair_e"),
                ("positron", 0.0, None, "pair_p")]},
            {"interaction": "pair", "secondaries": [
                ("electron", 5.0, None, "pair_e")]},
        ]
        result = summarize_interactions(records)
        self.assertEqual(result["polar_edges_deg"], list(range(0, 181, 10)))
        self.assertEqual(result["polar_counts"]["rayleigh"][0], 1)
        self.assertEqual(result["polar_counts"]["rayleigh"][-1], 1)
        self.assertEqual(result["polar_counts"]["compton"][1], 1)
        self.assertEqual(result["polar_counts"]["photo"][4], 1)
        self.assertEqual(sum(result["polar_counts"]["photo"]), 1)
        self.assertEqual(result["photo_shell_counts"]["O_K"], 1)
        self.assertEqual(sum(result["photo_shell_counts"].values()), 1)
        self.assertEqual(sum(result["pair_fraction_counts"]), 1)
        self.assertEqual(result["pair_fraction_counts"][22], 1)  # 3/(3+1)
        self.assertEqual(result["pair_fraction_undefined"], 2)


class ComparisonReportTests(unittest.TestCase):
    def test_older_saved_comparison_uses_available_data_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_comparison(directory, {"E2.0": {
                "MC1": _result(directory, "MC1", 2.0),
                "MC2": _result(directory, "MC2", 2.0),
                "BeamWeaver": {"skipped": "pair heads not ready"},
            }})
            output = directory / "figures"
            generated = regenerate_reports(directory, verbose=False)
            self.assertTrue((output / "pdd_E2.0.png").is_file())
            self.assertTrue((output / "interaction_fractions_E2.0.png").is_file())
            self.assertTrue((output / "timing_E2.0.png").is_file())
            self.assertTrue((output / "comparison_summary.csv").is_file())
            with (output / "comparison_summary.csv").open(newline="") as handle:
                table = {row["method"]: row for row in csv.DictReader(handle)}
            self.assertEqual(table["MC1"]["interaction_count"], "10")
            self.assertEqual(table["MC2"]["interaction_count"], "10")
            self.assertEqual(table["BeamWeaver"]["skipped_reason"],
                             "pair heads not ready")
            self.assertEqual(table["BeamWeaver"]["elapsed_wall_s"], "")
            self.assertEqual(table["BeamWeaver"]["photo_interaction_fraction"], "")
            self.assertFalse(list(output.glob("*angle_E2.0.png")))
            self.assertFalse(list(output.glob("pair_share_E2.0.png")))
            self.assertEqual(len(generated), len(set(generated)))

    def test_saved_event_counts_add_physical_figures_only_when_observed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            observed = summarize_interactions([
                {"interaction": "rayleigh", "angle": 5.0},
                {"interaction": "compton", "angle": 25.0},
                {"interaction": "photo", "angle": 45.0,
                 "secondaries": [("electron", 0.8, None, "photo_O_K")]},
                {"interaction": "pair", "secondaries": [
                    ("electron", 3.0, None, "pair_e"),
                    ("positron", 1.0, None, "pair_p")]},
            ])
            empty = summarize_interactions([])
            _write_comparison(directory, {
                "E2.0": {
                    "MC1": _result(directory, "MC1", 2.0, histograms=observed),
                    "MC2": _result(directory, "MC2", 2.0, histograms=empty),
                    "BeamWeaver": {"skipped": "pair heads not ready"},
                },
                "E0.5": {
                    "MC1": _result(directory, "MC1", 0.5, histograms=empty),
                    "MC2": _result(directory, "MC2", 0.5, histograms=empty),
                    "BeamWeaver": {"skipped": "checkpoint not available"},
                },
            })
            regenerate_reports(directory, verbose=False)
            output = directory / "figures"
            for figure in ("rayleigh_angle", "compton_angle",
                           "photoelectric_angle_shell", "pair_share"):
                self.assertTrue((output / f"{figure}_E2.0.png").is_file(), figure)
                self.assertTrue((output / f"{figure}_E2.0.pdf").is_file(), figure)
                self.assertFalse((output / f"{figure}_E0.5.png").exists(), figure)
            self.assertTrue((output / "pdd_E0.5.png").is_file())


if __name__ == "__main__":
    unittest.main()
