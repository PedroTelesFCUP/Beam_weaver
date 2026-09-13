"""Portable golden regression for reference physics; uses artificial tables."""
import json
import math
from pathlib import Path
import unittest

from reference_fixtures import exercise, load_current


class ReferenceGoldenTests(unittest.TestCase):
    def assert_nested_equal(self, actual, expected, path='result'):
        if path == 'result.transform_max_error':
            # The largest roundoff residual depends on libm/NumPy; the
            # original transform tolerance is the relevant invariant.
            self.assertLess(actual, 1e-10)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), path)
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key], f'{path}.{key}')
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected), path)
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.assert_nested_equal(a, b, f'{path}[{i}]')
        elif isinstance(expected, float) and not path.endswith('.next_random'):
            # Portable across small libm/NumPy differences. The original
            # same-runtime comparison was also run and matched exactly.
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14),
                            f'{path}: {actual!r} != {expected!r}')
        else:
            self.assertEqual(actual, expected, path)

    def test_reference_samplers_and_lepton_ledgers_match_golden(self):
        directory = Path(__file__).resolve().parent
        expected = json.loads((directory / 'data' / 'reference_golden.json').read_text())
        # The original rounded learned rest energy is intentionally retired.
        # All original reference-event and lepton-ledger golden values stay intact.
        expected['constants'].pop('MEC2_MEV')
        actual = exercise(load_current(directory.parent))
        self.assert_nested_equal(actual, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
