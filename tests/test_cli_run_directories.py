"""Automatic run names must be readable without reusing an existing run."""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import os
import tempfile
import unittest
from unittest.mock import patch

from beamweaver.cli import _new_run_dir, _record_command


class RunDirectoryTests(unittest.TestCase):
    def test_automatic_names_reserve_the_next_available_daily_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                with patch("beamweaver.cli.datetime") as clock:
                    clock.now.return_value = datetime(2026, 9, 23,
                                                       tzinfo=timezone.utc)
                    first = _new_run_dir(None, "compare")
                    (first / "comparison.json").write_text("preserve me")
                    second = _new_run_dir(None, "compare")
                    other = _new_run_dir(None, "generate")
                    self.assertEqual(first.name, "compare-20260923-v1")
                    self.assertEqual(second.name, "compare-20260923-v2")
                    self.assertEqual(other.name, "generate-20260923-v1")
                    self.assertEqual((first / "comparison.json").read_text(),
                                     "preserve me")
                    args = argparse.Namespace(command="compare", histories=2000)
                    _record_command(first, args)
                    _record_command(first, args)
                    self.assertEqual(len(list(first.glob("command-20260923-v*.json"))), 2)
            finally:
                os.chdir(original)

    def test_explicit_nonempty_output_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "chosen"
            self.assertEqual(_new_run_dir(output, "compare"), output.resolve())
            (output / "result.json").write_text("preserve me")
            with self.assertRaises(FileExistsError):
                _new_run_dir(output, "compare")
            self.assertEqual((output / "result.json").read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
