"""Current constants are recorded without inventing old training provenance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from beamweaver.constants import mec2, runtime_physics_metadata
from beamweaver.dataset import build_energy_grids, write_generator_spec

HAS_TORCH = importlib.util.find_spec('torch') is not None


class PhysicsMetadataTests(unittest.TestCase):
    def test_generator_spec_records_current_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'generator.json'
            specification = write_generator_spec(path)
            self.assertEqual(specification['physics_constants'], runtime_physics_metadata())
            recorded = specification['physics_constants']
            self.assertEqual(recorded['mec2_MeV'], mec2)
            self.assertEqual(recorded['pair_threshold_MeV'], 2. * mec2)
            self.assertEqual(recorded['hc_MeV_angstrom'], recorded['hc_keV_angstrom'] / 1000.)
            self.assertEqual(recorded['shell_bindings_eV']['O_K'], 532.)
            self.assertEqual(json.loads(path.read_text()), specification)
            self.assertEqual(specification['grids']['pair_threshold'], 2. * mec2)
            self.assertEqual(build_energy_grids()['pair_threshold'], 2. * mec2)

    def test_metadata_snapshots_do_not_share_mutable_storage(self):
        first = runtime_physics_metadata()
        expected = runtime_physics_metadata()
        self.assertIsNot(first, expected)
        self.assertEqual(first, expected)
        # Mutate every nested dictionary/list, then the outer snapshot.
        def mutate(value):
            if isinstance(value, dict):
                for child in list(value.values()):
                    mutate(child)
                value['test_only_mutation'] = True
            elif isinstance(value, list):
                value.append('test_only_mutation')
        mutate(first)
        self.assertEqual(runtime_physics_metadata(), expected)


@unittest.skipUnless(HAS_TORCH, 'Torch unavailable: training/provenance checks were not executed')
class TrainingPhysicsMetadataTests(unittest.TestCase):
    def test_training_masks_resolve_the_former_and_current_pair_thresholds(self):
        from beamweaver.constants import PROCESS_INDEX
        from beamweaver.training import factor_tensors
        from test_neural import edges_fixture
        old_threshold = 2. * .5109989
        threshold = 2. * mec2
        energies = [old_threshold, (old_threshold + threshold) / 2.,
                    np.nextafter(threshold, 0.), threshold,
                    np.nextafter(threshold, np.inf), 1.023]
        targets = {f'train|process|E{E:.17g}': np.array([1, 1, 1, int(E > threshold)])
                   for E in energies}
        self.assertEqual(len(targets), len(energies))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'threshold_targets.npz'
            np.savez(path, **targets)
            _, _, masks = factor_tensors(path, 'process', edges_fixture(), 'train')
        expected = [float(key.split('|E')[1]) > threshold for key in sorted(targets)]
        self.assertEqual(masks[:, PROCESS_INDEX['pair']].tolist(), expected)
        for name, index in PROCESS_INDEX.items():
            if name != 'pair':
                self.assertTrue(bool(masks[:, index].all()))

    def test_factor_checkpoint_records_current_training_constants(self):
        import torch
        from beamweaver.policy import GenerativeTransportPolicy
        from beamweaver.training import train_factor
        from test_neural import edges_fixture, write_targets
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'targets.npz'
            write_targets(path)
            model = GenerativeTransportPolicy(edges_fixture())
            train_factor(model, 'process', path, epochs=1, batch=3,
                         device='cpu', verbose=False, run_dir=root)
            saved = torch.load(root / 'v040_head_process.pt', weights_only=False)
            self.assertEqual(saved['training_physics_constants'], runtime_physics_metadata())

    def test_old_policy_load_does_not_rewrite_file_or_invent_training_provenance(self):
        import torch
        from beamweaver.policy import GenerativeTransportPolicy
        from test_neural import edges_fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path = root / 'existing_policy.pt'
            model = GenerativeTransportPolicy(edges_fixture())
            old_checkpoint = model.save_policy(root / 'template.pt', extra={'source_note': 'old fixture'})
            # Same current-schema weights/edges, saved before constants
            # provenance was recorded. Loading this case is still supported.
            old_checkpoint.pop('runtime_physics_constants', None)
            torch.save(old_checkpoint, old_path)
            before = hashlib.sha256(old_path.read_bytes()).hexdigest()
            loaded = GenerativeTransportPolicy.load_policy(old_path, device='cpu')
            self.assertEqual(hashlib.sha256(old_path.read_bytes()).hexdigest(), before)
            self.assertEqual(loaded.meta, {'source_note': 'old fixture'})
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)
            rewritten = loaded.save_policy(root / 'new_copy.pt')
            self.assertEqual(rewritten['runtime_physics_constants'], runtime_physics_metadata())
            self.assertNotIn('training_physics_constants', rewritten)
            self.assertNotIn('training_physics_constants', rewritten['meta'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
