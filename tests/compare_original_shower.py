"""Exact seeded original/cleaned MC shower regression with artificial tables.

Both runners use the cleaned WaterPhantom so this isolates transport and
its real imported reference functions. Geometry encoding is checked separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import pandas as pd

from reference_fixtures import load_original, plain, synthetic_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--package-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    original = load_original(args.source)
    sys.path.insert(0, str(args.package_dir.resolve()))
    from beamweaver.materials import WaterPhotonData
    from beamweaver.geometry import WaterPhantom
    from beamweaver.transport import run_reference_transport

    results = []
    with tempfile.TemporaryDirectory(prefix='beamweaver-showers-') as tmp:
        directory = Path(tmp)
        synthetic_tables(directory)
        # Enhance pairs in this artificial fixture to exercise annihilation
        # banking with few histories; this is not a physical water table.
        table = pd.read_csv(directory / 'Final_cross_sections.csv')
        table.loc[table.E >= 2., 'pair_triplet'] = 0.8
        table.to_csv(directory / 'Final_cross_sections.csv', index=False)
        pd.DataFrame({'E_MeV': [1e-6, 100.],
                      'S_col_MeV_per_cm': [20., 20.],
                      'S_rad_MeV_per_cm': [2., 2.]}
                    ).to_csv(directory / 'ElectronStoppingPower.csv', index=False)
        previous_directory = Path.cwd()
        os.chdir(directory)
        try:
            for E in [0.05, 2., 5.]:
                reports = []
                rng_states = []
                for data_type, runner in [(original.PenelopeLikeWaterData,
                                            original.run_mc_shower),
                                           (WaterPhotonData, run_reference_transport)]:
                    data = data_type('Final_cross_sections.csv', 'Rayleigh_cross_sections.csv')
                    env = WaterPhantom(data, fixed_energy=E)
                    random.seed(8128)
                    np.random.seed(8128)
                    result = runner(6, data, env, max_steps=10000, verbose=False)
                    reports.append(result)
                    rng_states.append((random.getstate(), np.random.get_state()))
                a, b = [json.dumps(plain(r), sort_keys=True, allow_nan=True) for r in reports]
                assert a == b, f'Original/cleaned shower results differ at {E} MeV'
                assert rng_states[0][0] == rng_states[1][0]
                np.testing.assert_array_equal(rng_states[0][1][1], rng_states[1][1][1])
                assert rng_states[0][1][0] == rng_states[1][1][0]
                assert rng_states[0][1][2:] == rng_states[1][1][2:]
                tracks, secondaries, dose, interactions = reports[0]
                entry = {'energy_MeV': E, 'histories': len(tracks),
                         'interactions': len(interactions),
                         'banked_photons': sum(len(t['banked_photons']) for t in tracks),
                         'pairs': sum(i['interaction'] == 'pair' for i in interactions),
                         'max_ledger_error_MeV': max(t['ledger']['closure_err'] for t in tracks),
                         'dose_sum_MeV': float(dose.sum()),
                         'exact_output_and_rng_match': True,
                         'result_sha256': hashlib.sha256(a.encode()).hexdigest()}
                results.append(entry)
        finally:
            os.chdir(previous_directory)
    assert sum(r['pairs'] for r in results) > 0, 'Fixture did not exercise pair creation'
    assert sum(r['banked_photons'] for r in results) > 0, 'Fixture did not exercise photon bank'
    report = {'fixture_kind': 'artificial tables, not physical validation',
              'cases': results, 'passed': True}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
