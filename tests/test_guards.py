"""Reference guard regression without Torch or material CSV dependencies."""
import sys
from pathlib import Path
from types import ModuleType
import unittest

PROJECT = Path(__file__).resolve().parents[1]
if not (PROJECT / 'beamweaver').is_dir():
    PROJECT = PROJECT / 'BeamWeaver'
sys.path.insert(0, str(PROJECT))
from beamweaver import audit, dataset, materials, physics


class GuardTests(unittest.TestCase):
    def targets(self):
        return [(physics, 'sample_rayleigh_event'), (physics, 'sample_compton_event'),
                (physics, 'sample_photoelectric_event'),
                (physics, 'sample_photoelectric_event_for_shell'),
                (physics, 'sample_pair_event'), (physics, 'sample_photon_interaction'),
                (dataset, 'sample_rayleigh_event'), (dataset, 'sample_compton_event'),
                (dataset, 'sample_photoelectric_event_for_shell'),
                (dataset, 'sample_pair_event'),
                (physics.ComptonEnergyTransferSampler, 'sample_energy_transfer'),
                (materials.WaterPhotoShellData, 'sample_shell'),
                (materials.WaterPhotonData, 'sample_photo_shell_index')]

    def test_blocks_actual_module_aliases_and_methods_and_restores(self):
        targets = self.targets()
        original = [getattr(owner, name) for owner, name in targets]
        evidence = audit.ExecutionAudit()
        with audit.reference_sampler_guard(evidence):
            for owner, name in targets:
                # No arguments: the tripwire must fire before real sampler
                # code can inspect inputs or attempt to load a CSV.
                with self.subTest(owner=owner, name=name):
                    with self.assertRaisesRegex(RuntimeError, 'reference sampler'):
                        getattr(owner, name)()
        self.assertEqual(evidence.reference_sampler_calls, len(targets))
        self.assertEqual(evidence.analytic_shell_calls, 2)
        self.assertEqual(evidence.reference_guard_entries, 1)
        self.assertEqual(evidence.reference_guard_exits, 1)
        for (owner, name), function in zip(targets, original):
            self.assertIs(getattr(owner, name), function)

    def test_exception_restores_aliases_and_releases_guard(self):
        original = physics.sample_rayleigh_event
        evidence = audit.ExecutionAudit()
        with self.assertRaisesRegex(ValueError, 'fixture interruption'):
            with audit.reference_sampler_guard(evidence):
                raise ValueError('fixture interruption')
        self.assertIs(physics.sample_rayleigh_event, original)
        with audit.reference_sampler_guard(evidence):
            pass
        self.assertEqual(evidence.reference_guard_entries, 2)
        self.assertEqual(evidence.reference_guard_exits, 2)

    def test_nested_guard_rejected_without_disabling_outer_guard(self):
        outer, inner = audit.ExecutionAudit(), audit.ExecutionAudit()
        with audit.reference_sampler_guard(outer):
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                with audit.reference_sampler_guard(inner):
                    self.fail('nested process-wide guard unexpectedly entered')
            with self.assertRaisesRegex(RuntimeError, 'reference sampler'):
                physics.sample_pair_event()
        self.assertEqual(outer.reference_sampler_calls, 1)
        self.assertEqual(outer.reference_guard_entries, outer.reference_guard_exits)
        self.assertEqual(inner.reference_guard_entries, 0)

    def test_alias_imported_during_guard_is_restored(self):
        module_name = 'beamweaver._guard_late_fixture'
        self.assertNotIn(module_name, sys.modules)
        original = physics.sample_pair_event
        evidence = audit.ExecutionAudit()
        try:
            with audit.reference_sampler_guard(evidence):
                late_module = ModuleType(module_name)
                sys.modules[module_name] = late_module
                exec('from beamweaver.physics import sample_pair_event as imported_pair',
                     vars(late_module))
                with self.assertRaisesRegex(RuntimeError, 'reference sampler'):
                    late_module.imported_pair()
            self.assertIs(late_module.imported_pair, original)
            self.assertEqual(evidence.reference_sampler_calls, 1)
        finally:
            sys.modules.pop(module_name, None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
