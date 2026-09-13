#!/usr/bin/env python3
"""Focused structural-cleanup regression; synthetic inputs are NOT water data.

No Torch mocks: original NumPy definitions are compiled directly from its AST.
The same fixture runner can be applied to an imported cleaned implementation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd


DEFINITIONS = {
    'WaterPhotoShellData', 'ComptonSampler', 'PenelopeLikeWaterData',
    'rotate_direction', 'sample_rayleigh', 'sample_compton',
    '_sauter_dcs_shape', '_sauter_envelope_max',
    'sample_photoelectric_forced', 'sample_photoelectric',
    '_pair_fC', '_pair_F0', '_pair_g1g2', '_pair_phi12',
    '_pair_sample_eps', '_pair_lepton_costheta', 'sample_pair',
    'photon_interact', 'stopping_power', 'transport_lepton_csda',
    'load_stopping_power', 'run_mc_shower',
    'bw4_norm_logE', 'bw4_k_invA', 'bw4_q2max', 'bw4_q2_from_cos',
    'bw4_cos_from_q2', 'bw4_lq2r_from_cos', 'bw4_cos_from_lq2r',
    'bw4_tau_min', 'bw4_u_from_tau', 'bw4_tau_from_u',
    'bw4_cos_from_tau', 'bw4_transform_selftest', 'bw4_grids', '_bw4_seed',
}
CONSTANTS = {
    'mec2', 'MEC2_MEV', 'HC_KEV_A', 'HC_MEV_A',
    'PHOTO_SHELL_BINDINGS', '_SAUTER_ENV_CACHE', '_PAIR_ALPHA_FS',
    '_PAIR_R_TABLE', '_PAIR_Z_EQ', '_PAIR_R_EQ',
    'BW4_EMIN_MEV', 'BW4_EMAX_MEV', 'BW4_PAIR_TRAIN_EXTRA', 'BW4_TEST_E',
}

# Original-source labels are retained only inside this regression harness.
# The current package exposes the descriptive APIs on the right.
CURRENT_NAMES = {
    'PHOTO_SHELL_BINDINGS': 'PHOTO_SHELL_BINDINGS_EV',
    'ComptonSampler': 'ComptonEnergyTransferSampler',
    'PenelopeLikeWaterData': 'WaterPhotonData',
    'sample_rayleigh': 'sample_rayleigh_event',
    'sample_compton': 'sample_compton_event',
    'sample_photoelectric': 'sample_photoelectric_event',
    'sample_photoelectric_forced': 'sample_photoelectric_event_for_shell',
    'sample_pair': 'sample_pair_event',
    'photon_interact': 'sample_photon_interaction',
    'bw4_grids': 'build_energy_grids',
    '_bw4_seed': 'deterministic_seed',
    'bw4_norm_logE': 'norm_logE',
    'bw4_k_invA': 'k_invA',
    'bw4_q2max': 'q2max',
    'bw4_q2_from_cos': 'q2_from_cos',
    'bw4_cos_from_q2': 'cos_from_q2',
    'bw4_lq2r_from_cos': 'lq2r_from_cos',
    'bw4_cos_from_lq2r': 'cos_from_lq2r',
    'bw4_tau_min': 'tau_min',
    'bw4_u_from_tau': 'u_from_tau',
    'bw4_tau_from_u': 'tau_from_u',
    'bw4_cos_from_tau': 'cos_from_tau',
    'bw4_transform_selftest': 'transform_selftest',
}


def load_original(path: Path) -> SimpleNamespace:
    """Execute selected source definitions, excluding imports and main code."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in DEFINITIONS:
                selected.append(node)
        elif isinstance(node, ast.Assign):
            names = {n.id for target in node.targets for n in ast.walk(target)
                     if isinstance(n, ast.Name)}
            if names & CONSTANTS:
                selected.append(node)
    namespace = {
        'np': np, 'pd': pd, 'math': math, 'os': os, 'random': random,
        '_v4hash': hashlib,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'),
         namespace)
    missing = (DEFINITIONS | CONSTANTS) - namespace.keys()
    if missing:
        raise AssertionError(f'Missing original definitions: {sorted(missing)}')
    return SimpleNamespace(**namespace)


def load_current(package_dir: Path) -> SimpleNamespace:
    """Expose current implementations under fixture labels for comparison."""
    sys.path.insert(0, str(package_dir.resolve()))
    namespace = {}
    for module in ('constants', 'materials', 'physics', 'coordinates', 'dataset'):
        namespace.update(vars(importlib.import_module('beamweaver.' + module)))
    for original_name, current_name in CURRENT_NAMES.items():
        namespace[original_name] = namespace[current_name]
    return SimpleNamespace(**namespace)


def synthetic_tables(directory: Path):
    """Tiny smooth artificial tables solely for comparing code paths."""
    E = np.array([0.001, 0.01, 0.1, 0.5, 1.022, 2.0, 10.0])
    tables = {
        'Final_cross_sections.csv': {
            'E': E, 'photoelectric': 0.01 / E,
            'compton': np.full_like(E, 0.4),
            'pair_triplet': np.array([0., 0., 0., 0., 0., 0.05, 0.1]),
        },
        'Rayleigh_cross_sections.csv': {'E': E, 'coh': 0.001 / E},
        'WaterPhotoShells.csv': {
            'E_MeV': E,
            **{name + '_cm2g': (i + 1) * 0.01 / E for i, name in enumerate(
                ['H_K', 'O_K', 'O_L1', 'O_L2', 'O_L3'])},
        },
        'water_fq.csv': {
            'q': np.r_[0., np.geomspace(0.001, 1000., 70)],
            'F_q': 10. / (1. + np.r_[0., np.geomspace(0.001, 1000., 70)]**2),
        },
        'water_sq.csv': {'q': [0., 1., 1000.], 'S_q': [10., 10., 10.]},
    }
    for name, columns in tables.items():
        pd.DataFrame(columns).to_csv(directory / name, index=False)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def exercise(implementation: SimpleNamespace) -> dict:
    """Execute actual loaded functions and verify physical/interface invariants."""
    m = implementation
    report = {'fixture_kind': 'artificial, not a water-physics validation'}
    report['transform_max_error'] = m.bw4_transform_selftest()
    grids = m.bw4_grids()
    report['grid_hash'] = grids['sha256']
    assert (len(grids['train']), len(grids['val']), len(grids['test'])) == (69, 63, 9)
    assert grids['pair_threshold'] == (m.PAIR_THRESHOLD_MEV
        if hasattr(m, 'PAIR_THRESHOLD_MEV') else 2. * m.MEC2_MEV)
    report['constants'] = {name: getattr(m, name) for name in
                           ['mec2', 'MEC2_MEV', 'HC_KEV_A', 'HC_MEV_A'] if hasattr(m, name)}

    grid, values = np.array([1., 2., 4., 8.]), np.array([0., 0., 2., 8.])
    energies = np.array([0.5, 1., 1.5, 2., 3., 4., 6., 8., 12.])
    shell = m.WaterPhotoShellData.__new__(m.WaterPhotoShellData)
    water = m.PenelopeLikeWaterData.__new__(m.PenelopeLikeWaterData)
    scalar = np.array([water.loglog_interp(E, grid, values) for E in energies])
    other_scalar = np.array([shell._loglog_interp(E, grid, values) for E in energies])
    np.testing.assert_array_equal(other_scalar, scalar)
    assert np.all(scalar[:5] == 0.)
    report['interpolation'] = plain(scalar)

    directions = ([0., 0., 1.], [0., 0., -1.], [2., -1., 3.], [0., 0., 0.])
    rotations = []
    for d in directions:
        for theta in (0., 0.7, math.pi):
            out = m.rotate_direction(d, theta, 1.2)
            np.testing.assert_allclose(np.linalg.norm(out), 1., atol=3e-15)
            rotations.append(out)
    report['rotations'] = plain(rotations)

    with tempfile.TemporaryDirectory(prefix='beamweaver-synthetic-') as tmp:
        directory = Path(tmp)
        synthetic_tables(directory)
        data = m.PenelopeLikeWaterData(
            str(directory / 'Final_cross_sections.csv'),
            str(directory / 'Rayleigh_cross_sections.csv'),
            water_shell_csv=str(directory / 'WaterPhotoShells.csv'),
            coherent_ff_csv=str(directory / 'water_fq.csv'))
        results = {}
        for seed, E in enumerate((0.001, 0.05, 0.5, 2., 10.), 100):
            comp = m.ComptonSampler(E, sq_csv=str(directory / 'water_sq.csv'))
            for kind in ('rayleigh', 'compton', 'photo', 'pair'):
                random.seed(seed)
                np.random.seed(seed)
                events = []
                for _ in range(5):
                    if kind == 'compton':
                        event = m.sample_compton(E, [0., 0., 1.], data, sampler=comp)
                    else:
                        sampler = getattr(m, {'photo': 'sample_photoelectric'}.get(kind,
                                          'sample_' + kind))
                        event = sampler(E, [0., 0., 1.], data)
                    outgoing, Eout, secondaries, process, shell_onehot = event
                    if process == 'rayleigh':
                        assert Eout == E and not secondaries
                    elif process == 'compton':
                        np.testing.assert_allclose(Eout + secondaries[0][1], E,
                                                   atol=2e-15, rtol=2e-15)
                    elif process == 'pair':
                        kinetic = sum(particle[1] for particle in secondaries)
                        np.testing.assert_allclose(kinetic + 2. * m.mec2, E,
                                                   atol=2e-15, rtol=2e-15)
                    elif process == 'photo':
                        shell_name = ['H_K', 'O_K', 'O_L1', 'O_L2', 'O_L3'][
                            shell_onehot.index(1)]
                        np.testing.assert_allclose(
                            secondaries[0][1] + m.PHOTO_SHELL_BINDINGS[shell_name] * 1e-6,
                            E, atol=2e-15, rtol=2e-15)
                    if Eout > 0 and process != 'pair_subthresh':
                        np.testing.assert_allclose(np.linalg.norm(outgoing), 1., atol=5e-15)
                    events.append(event)
                results[f'{kind}@{E:g}'] = {'events': plain(events),
                                           'next_random': random.random()}
        report['samplers'] = results

    env = SimpleNamespace(xmin=-1., xmax=1., ymin=-1., ymax=1.,
                          zmin=0., zmax=2., dz=0.01, pdd_bins=200)
    Egrid = np.array([1e-6, 100.])
    stopping_col = np.array([1., 1.])
    stopping_rad = np.array([0.1, 0.1])
    transport = {}
    cases = {
        'stops_inside': (0.01, [0., 0., 1.], [0., 0., 0.5]),
        'boundary_escape': (0.01, [1., 0., 0.], [0.999, 0., 0.5]),
        'starts_outside': (0.01, [1., 0., 0.], [1.1, 0., 0.5]),
        'below_cutoff': (0.0005, [0., 0., 1.], [0., 0., 0.5]),
        'subcutoff_escape': (0.0012, [1., 0., 0.], [0.9996, 0., 0.5]),
        'radiative_budget': (0.1, [0., 0., 1.], [0., 0., 0.5]),
    }
    for name, (E, d, p) in cases.items():
        for positron in (False, True):
            random.seed(2040)
            dose = np.zeros(env.pdd_bins)
            result = m.transport_lepton_csda(E, d, p, dose, env,
                Egrid=Egrid, S_col_vals=stopping_col, S_rad_vals=stopping_rad,
                is_positron=positron)
            deposited, secondaries, ledger = result
            np.testing.assert_allclose(deposited + ledger['brems_energy'] +
                                       ledger['escaped_kinetic'], E, atol=1e-12)
            np.testing.assert_allclose(dose.sum(), deposited, atol=1e-12)
            if name in ('boundary_escape', 'starts_outside', 'subcutoff_escape'):
                assert ledger['escaped']
                assert ledger['cutoff_deposit'] == 0.
                if positron:
                    assert ledger['annih_energy'] == 0.
                    assert ledger['annih_potential_escaped'] == 2. * m.mec2
            elif positron:
                photons = [s for s in secondaries if s[3] == 'annih']
                assert len(photons) == 2
                np.testing.assert_array_equal(photons[0][2], -photons[1][2])
                assert photons[0][1] == photons[1][1] == m.mec2
            transport[f'{name}:{"positron" if positron else "electron"}'] = {
                'result': plain(result), 'dose': plain(dose), 'next_random': random.random()}
    report['lepton_transport'] = transport
    report['passed'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--package-dir', type=Path,
                        help='Import beamweaver here and compare to original outputs exactly')
    args = parser.parse_args()
    results = exercise(load_original(args.source))
    if args.package_dir:
        actual = exercise(load_current(args.package_dir))
        assert actual == results, 'Package differs from original fixture outputs or RNG state'
        results['package_exact_equivalence'] = True
    results['source_sha256'] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if args.out:
        args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': results['passed'],
                      'coordinate_max_abs_error': results['transform_max_error'],
                      'sampler_cases': len(results['samplers']),
                      'lepton_cases': len(results['lepton_transport']),
                      'grid_hash': results['grid_hash']}))


if __name__ == '__main__':
    main()
