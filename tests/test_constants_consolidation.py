"""Contracts at the boundaries touched by consolidation of shared constants.

Artificial tables exercise interfaces and reproducibility, not water accuracy.
"""
import math
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from reference_fixtures import synthetic_tables
from beamweaver.geometry import WaterPhantom, encode_policy_energy, encode_policy_energy_batch
from beamweaver.materials import WaterPhotonData
from beamweaver.physics import rotate_direction, sample_photoelectric_event_for_shell


class ConstantsConsolidationTests(unittest.TestCase):
    def test_local_frame_inverse_across_both_polar_branches(self):
        env = WaterPhantom(None)
        for uz in (-1., -.99990001, -.99989999, .3, .99989999, .99990001, 1.):
            transverse = math.sqrt(1. - uz * uz)
            incident = np.array([.6 * transverse, .8 * transverse, uz])
            for phi in (.1, 1.7, 5.8):
                scattered = rotate_direction(incident, .9, phi)
                recovered = env._extract_local_phi(incident, scattered, .9)
                error = (recovered - phi + math.pi) % (2. * math.pi) - math.pi
                self.assertAlmostEqual(error, 0., places=12)
                self.assertAlmostEqual(float(np.linalg.norm(scattered)), 1., places=14)

    def test_reset_preserves_source_draws_and_caller_settings(self):
        env = WaterPhantom(None, fixed_energy=.3, ecut=.002, energy_range=(.01, 5.))
        expected_rng = random.Random(9182)
        expected_position = tuple(expected_rng.random() * 10. - 5. for _ in range(2)) + (0.,)
        env.reset(seed=9182)
        self.assertEqual(env.initial_position, expected_position)
        self.assertEqual(random.random(), expected_rng.random())
        self.assertEqual((env.E, env.ecut, env.E_min, env.E_max), (.3, .002, .01, 5.))
        self.assertEqual(env.dz, (env.zmax - env.zmin) / env.pdd_bins)

    def test_scalar_and_batch_energy_encoding_preserve_distinct_rounding(self):
        energies = np.r_[[-1., 0., .001, .01, 5., 10.], np.geomspace(.0001, 100., 20000)]
        for energy_range in ((.001, 10.), (.01, 5.), (.0001, 20.)):
            env = WaterPhantom(None, energy_range=energy_range)
            lo, hi = map(math.log10, energy_range)
            # Frozen observation-encoding contract; scalar and vector paths
            # use their original different log10 implementations.
            scalar = [10. ** float(np.float32(np.clip(math.log10(max(E, .001, env.E_min)), lo, hi)))
                      for E in energies]
            vector = 10. ** np.clip(np.log10(np.maximum(np.maximum(energies, .001), env.E_min)),
                                   lo, hi).astype(np.float32).astype(np.float64)
            np.testing.assert_array_equal([encode_policy_energy(E, env) for E in energies], scalar)
            np.testing.assert_array_equal(encode_policy_energy_batch(energies, env), vector)

    def test_shell_categories_keep_order_and_return_independent_vectors(self):
        names = ['H_K', 'O_K', 'O_L1', 'O_L2', 'O_L3']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            synthetic_tables(path)
            data = WaterPhotonData(str(path / 'Final_cross_sections.csv'),
                                   str(path / 'Rayleigh_cross_sections.csv'),
                                   water_shell_csv=str(path / 'WaterPhotoShells.csv'),
                                   coherent_ff_csv=str(path / 'water_fq.csv'))
            for shell, name in enumerate(names):
                first = sample_photoelectric_event_for_shell(1., [0., 0., 1.], data, shell)
                second = sample_photoelectric_event_for_shell(1., [0., 0., 1.], data, shell)
                expected = [int(i == shell) for i in range(5)]
                self.assertEqual(first[4], expected)
                self.assertEqual(second[4], expected)
                self.assertEqual(first[2][0][3], 'photo_' + name)
                first[4][shell] = 0
                self.assertEqual(second[4], expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
