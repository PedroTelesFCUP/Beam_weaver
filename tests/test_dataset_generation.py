"""Exercise dataset generation and saved interpretation using artificial tables."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reference_fixtures import synthetic_tables
from beamweaver.dataset import generate_dataset, write_generator_spec
from beamweaver.materials import WaterPhotonData


class DatasetGenerationTests(unittest.TestCase):
    def test_generation_preserves_split_edges_and_reports_skipped_checks(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            synthetic_tables(target)
            try:
                os.chdir(target)
                data = WaterPhotonData('Final_cross_sections.csv', 'Rayleigh_cross_sections.csv')
                output = target / 'dataset.npz'
                with redirect_stdout(io.StringIO()):
                    manifest = generate_dataset(data, out=str(output), grid_stride=8,
                        M={name: 64 for name in ('process', 'shell', 'ray', 'comp', 'photo', 'pair')},
                        verbose=False)
                    write_generator_spec(str(target / 'spec.json'))
                with np.load(output) as saved:
                    metadata = json.loads(saved['__bw4_meta__'].tobytes())
                    self.assertEqual(metadata['schema_version'], 4)
                    self.assertEqual(set(metadata['grids']), {'train', 'val', 'test'})
                    self.assertEqual(metadata['M']['comp'], 64)
                    for name, entry in manifest['edges'].items():
                        edges = saved['__edges__|' + name]
                        self.assertTrue(np.all(np.diff(edges) > 0))
                        self.assertEqual(entry['sha256'], hashlib.sha256(edges.tobytes()).hexdigest())
                    self.assertTrue(any(k.startswith('test|') for k in saved.files))
                self.assertEqual(manifest['npz_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
                checks = manifest['sanity_check']
                self.assertTrue(any(r['insufficient_statistics'] for r in checks.values()))
                for result in checks.values():
                    if result['insufficient_statistics']:
                        self.assertIsNone(result['pass'])
                        self.assertIsNone(result['worst_floor'])
                self.assertTrue((target / 'dataset_manifest.json').is_file())
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main(verbosity=2)
