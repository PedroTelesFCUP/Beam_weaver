"""Compact plots-ready summaries of recorded recursive photon interactions.

Each record belongs to one collision in a transport history.  The records
include collisions of banked secondary photons, so their incident energies
generally differ from the initial primary photon energy of the comparison.
"""

import math

from .constants import SHELL_NAMES


def _bin_index(value, upper, n_bins):
    """Use half-open bins, except that the last bin includes ``upper``."""
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(x) or not 0.0 <= x <= upper:
        return None
    return min(int(x * n_bins / upper), n_bins - 1)


def _secondary_tagged_kinetic(secondary, species, tag):
    if not isinstance(secondary, (tuple, list)) or len(secondary) < 4:
        return None
    if secondary[0] != species or secondary[3] != tag:
        return None
    try:
        energy = float(secondary[1])
    except (TypeError, ValueError, OverflowError):
        return None
    return energy if math.isfinite(energy) and energy >= 0.0 else None


def summarize_interactions(records):
    """Bin event outcomes directly from MC/learned recursive shower records.

    Rayleigh and Compton angles are scattered-photon polar angles; photo angles
    are photoelectron polar angles.  Pair fractions are electron kinetic energy
    divided by the sum of electron and positron kinetic energies for the *same*
    pair event.  Zero-denominator or incomplete pair records are tracked rather
    than assigned a fabricated fraction.
    """
    summary = {
        "polar_edges_deg": list(range(0, 181, 10)),
        "polar_counts": {name: [0] * 18 for name in
                         ("rayleigh", "compton", "photo")},
        "photo_shell_counts": {name: 0 for name in SHELL_NAMES},
        "pair_fraction_edges": [i / 30 for i in range(31)],
        "pair_fraction_counts": [0] * 30,
        "pair_fraction_undefined": 0,
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        process = record.get("interaction")
        secondaries = record.get("secondaries") or ()
        if process in summary["polar_counts"]:
            idx = _bin_index(record.get("angle"), 180.0, 18)
            if idx is not None:
                summary["polar_counts"][process][idx] += 1
        if process == "photo":
            shell_tags = [sec[3][len("photo_"):]
                          for sec in secondaries
                          if isinstance(sec, (tuple, list)) and len(sec) >= 4
                          and sec[0] == "electron"
                          and isinstance(sec[3], str)
                          and sec[3].startswith("photo_")]
            if len(shell_tags) == 1 and shell_tags[0] in summary["photo_shell_counts"]:
                summary["photo_shell_counts"][shell_tags[0]] += 1
        elif process == "pair":
            electrons = [_secondary_tagged_kinetic(sec, "electron", "pair_e")
                         for sec in secondaries]
            positrons = [_secondary_tagged_kinetic(sec, "positron", "pair_p")
                         for sec in secondaries]
            electrons = [e for e in electrons if e is not None]
            positrons = [e for e in positrons if e is not None]
            if len(electrons) != 1 or len(positrons) != 1:
                summary["pair_fraction_undefined"] += 1
                continue
            kinetic_sum = electrons[0] + positrons[0]
            if not math.isfinite(kinetic_sum) or kinetic_sum <= 0.0:
                summary["pair_fraction_undefined"] += 1
                continue
            idx = _bin_index(electrons[0] / kinetic_sum, 1.0, 30)
            if idx is None:
                summary["pair_fraction_undefined"] += 1
            else:
                summary["pair_fraction_counts"][idx] += 1
    return summary
