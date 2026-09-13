"""Small neural integration checks; explicitly skipped when Torch is absent.

Run with the cleaned package on PYTHONPATH. To compare against the uploaded
monolith too, set BEAMWEAVER_ORIGINAL to that Python file's absolute path.
Artificial categorical targets test execution, never physical accuracy.
"""
from dataclasses import asdict
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if not (PROJECT / 'beamweaver').is_dir():
    PROJECT = PROJECT / 'BeamWeaver'
sys.path.insert(0, str(PROJECT))
HAS_TORCH = importlib.util.find_spec('torch') is not None


def edges_fixture():
    return {'ray_q2': np.array([-12., -3., 0.]),
            'comp_u': np.array([0., 0.5, 1.]),
            'photo_nu': np.array([0., 1., 2.]),
            'phi': np.array([0., np.pi, 2. * np.pi]),
            'pair_frac': np.array([0., 0.5, 1.]),
            'pair_nu_e': np.array([0., 1., 2.]),
            'pair_nu_p': np.array([0., 1., 2.])}


def write_targets(path):
    arrays = {}
    for role, E in [('train', 2.), ('val', 3.)]:
        tag = f'{role}|process|E{E}'
        arrays[tag] = np.array([1, 2, 3, 4])
        arrays[f'{role}|shell|E{E}'] = np.array([1, 2, 3, 4, 5])
        for fac, field, values in [('ray', 'lq2r', [-10., -9., -8., -2.]),
                                   ('comp', 'u', [0.1, 0.2, 0.3, 0.9])]:
            base = f'{role}|{fac}|E{E}'
            arrays[base + '|' + field] = np.array(values)
            arrays[base + '|phi'] = np.array([0.2, 0.3, 0.4, 4.])
            arrays[base + '|phi_valid'] = np.ones(4, dtype=np.uint8)
        for shell in range(5):
            base = f'{role}|photo|E{E}|H{shell}'
            arrays[base + '|nu'] = np.array([0.1, 0.2, 0.3, 1.5])
            arrays[base + '|phi'] = np.array([0.2, 0.3, 0.4, 4.])
            arrays[base + '|phi_valid'] = np.ones(4, dtype=np.uint8)
        base = f'{role}|pair|E{E}'
        arrays[base + '|f'] = np.array([0.1, 0.2, 0.3, 0.8])
        for lepton in ['e', 'p']:
            arrays[base + '|nu_' + lepton] = np.array([0.2, 0.3, 0.4, 1.5])
            arrays[base + '|phi_' + lepton] = np.array([0.2, 0.3, 0.4, 4.])
            arrays[base + '|phi_' + lepton + '_valid'] = np.ones(4, dtype=np.uint8)
    np.savez(path, **arrays)


@unittest.skipUnless(HAS_TORCH, 'Torch unavailable: neural checks were not executed')
class NeuralIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from beamweaver.constants import BW4_FACTOR_HEADS
        from beamweaver.policy import GenerativeTransportPolicy
        from beamweaver.training import train_factor
        cls.torch = torch
        cls.temp = tempfile.TemporaryDirectory(prefix='beamweaver-neural-fixture-')
        cls.directory = Path(cls.temp.name)
        cls.dataset = cls.directory / 'artificial_targets.npz'
        write_targets(cls.dataset)
        cls.model = GenerativeTransportPolicy(edges_fixture())
        cls.initial = {k: v.detach().clone() for k, v in cls.model.state_dict().items()}
        cls.losses = {}
        # Exercises both grouped CE and per-event pair CE, target building,
        # isolation assertion, best-state restore and checkpoint emission.
        for factor in BW4_FACTOR_HEADS:
            cls.losses[factor] = train_factor(
                cls.model, factor, str(cls.dataset), epochs=1, batch=3,
                device='cpu', verbose=False, run_dir=cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_every_head_trains_and_saves_with_finite_loss(self):
        self.assertEqual(len(self.losses), 13)
        self.assertTrue(all(np.isfinite(x) for x in self.losses.values()))
        self.assertTrue(all(self.model.ready.values()))
        self.assertTrue(self.model.pair_heads_ready)
        self.assertTrue(any(not self.torch.equal(v, self.initial[k])
                            for k, v in self.model.state_dict().items()))
        for factor in self.losses:
            self.assertTrue((self.directory / f'v040_head_{factor}.pt').is_file())

    def test_checkpoint_roundtrip_preserves_scalar_and_batch_draws(self):
        from beamweaver.policy import GenerativeTransportPolicy
        path = self.directory / 'policy.pt'
        self.model.save_policy(path)
        restored = GenerativeTransportPolicy.load_policy(path, device='cpu')
        self.assertEqual(restored.ready, self.model.ready)
        for key, value in self.model.state_dict().items():
            self.assertTrue(self.torch.equal(value, restored.state_dict()[key]), key)
        for E in [0.001, 0.1, 2., 10.]:
            a = self.model.generate_event(E, self.torch.Generator().manual_seed(123))
            b = restored.generate_event(E, self.torch.Generator().manual_seed(123))
            self.assertEqual(asdict(a), asdict(b))
        energies = np.array([0.001, 0.1, 2., 10.] * 16)
        a = self.model.generate_events_batch(energies,
                    generator=self.torch.Generator().manual_seed(123))
        b = restored.generate_events_batch(energies,
                    generator=self.torch.Generator().manual_seed(123))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_readiness_and_pair_support(self):
        from beamweaver.constants import PAIR_THRESHOLD_MEV
        from beamweaver.policy import GenerativeTransportPolicy
        untrained = GenerativeTransportPolicy(edges_fixture())
        with self.assertRaisesRegex(RuntimeError, 'process'):
            untrained.generate_event(1.)
        E = np.repeat([0.001, PAIR_THRESHOLD_MEV,
                       np.nextafter(PAIR_THRESHOLD_MEV, 0.)], 128)
        result = self.model.generate_events_batch(E,
                    generator=self.torch.Generator().manual_seed(900))
        self.assertFalse(np.any(result['process'] == 3))
        self.assertTrue(np.all(np.isnan(result['pair_frac'])))
        self.assertTrue(np.all(np.isfinite(result['theta_deg'])))
        self.assertTrue(np.all((result['theta_deg'] >= 0.) &
                               (result['theta_deg'] <= 180.)))

    @unittest.skipUnless(os.environ.get('BEAMWEAVER_ORIGINAL'),
                         'Set BEAMWEAVER_ORIGINAL for original-source neural parity')
    def test_original_seeded_scalar_and_batch_parity(self):
        source = Path(os.environ['BEAMWEAVER_ORIGINAL'])
        spec = importlib.util.spec_from_file_location('beamweaver_original_regression', source)
        original_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = original_module
        spec.loader.exec_module(original_module)
        original = original_module.GenerativeTransportPolicy(edges_fixture())
        for key, value in original.state_dict().items():
            self.assertTrue(self.torch.equal(value, self.initial[key]), key)
        # One grouped factor and one per-event factor compare the actual
        # original trainer numerics to this test's cleaned training results.
        previous_directory = Path.cwd()
        original_run = self.directory / 'original-training'
        original_run.mkdir(exist_ok=True)
        os.chdir(original_run)
        try:
            for factor in ['process', 'pair_e_polar']:
                loss = original_module.bw4_train_factor(
                    original, factor, str(self.dataset), epochs=1, batch=3,
                    device='cpu', verbose=False)
                self.assertEqual(loss, self.losses[factor])
                attr = original_module.BW4_FACTOR_HEADS[factor][0]
                for name, value in getattr(original, attr).state_dict().items():
                    self.assertTrue(self.torch.equal(
                        value, getattr(self.model, attr).state_dict()[name]),
                        f'{factor}.{name}')
        finally:
            os.chdir(previous_directory)
        original.load_state_dict(self.model.state_dict())
        original.ready = dict(self.model.ready)
        energies = np.array([0.001, 0.1, 2., 10.] * 16)
        obs = self.torch.zeros((len(energies), 5), dtype=self.torch.float32)
        obs[:, 4] = self.torch.tensor(np.log10(energies), dtype=self.torch.float32)
        decoded = 10. ** np.asarray(obs[:, 4], dtype=np.float64)
        for E in decoded[:4]:
            ga = self.torch.Generator().manual_seed(808)
            gb = self.torch.Generator().manual_seed(808)
            a = original.generate_event(float(E), ga)
            b = self.model.generate_event(float(E), gb)
            self.assertEqual(asdict(a), asdict(b))
            self.assertTrue(self.torch.equal(ga.get_state(), gb.get_state()))
        ga = self.torch.Generator().manual_seed(809)
        gb = self.torch.Generator().manual_seed(809)
        a = original.sample_pretrained_transport_events(obs, generator=ga)
        b = self.model.generate_events_batch(decoded, generator=gb)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(self.torch.equal(ga.get_state(), gb.get_state()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
