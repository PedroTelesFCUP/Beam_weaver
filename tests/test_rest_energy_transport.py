"""Rest-energy invariants through actual reference and learned transport code.

The policy is deterministic and the attenuation/stopping tables are artificial.
These checks cover event reconstruction and energy accounting, not the physical
accuracy of material tables or a trained neural kernel. Both learned runners
execute their real Compton/pair branches and the shared lepton transport.
"""

import math
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if not (PROJECT / "beamweaver").is_dir():
    PROJECT = PROJECT / "BeamWeaver"
sys.path.insert(0, str(PROJECT))

from beamweaver.events import PhotonCollisionEvent
from beamweaver.geometry import WaterPhantom
from beamweaver.physics import sample_pair_event
from beamweaver.transport import (
    run_learned_transport,
    run_learned_transport_batched,
)


class _FixedCollisionPolicy:
    pair_heads_ready = True

    def __init__(self, event):
        self.event = event

    def generate_event(self, energy, generator=None, audit=None):
        return self.event

    def generate_events_batch(self, energies, generator=None, audit=None):
        count = len(energies)
        event = self.event
        result = {
            "process": np.full(count, event.process, dtype=np.int64),
            "shell": np.full(count, event.shell, dtype=np.int64),
            "theta_deg": np.full(count, event.theta_deg, dtype=float),
            "phi_deg": np.full(count, event.phi_deg, dtype=float),
        }
        pair = event.pair if event.pair is not None else (math.nan,) * 5
        for name, value in zip(
            ("pair_frac", "pair_e_theta", "pair_e_phi", "pair_p_theta", "pair_p_phi"),
            pair,
        ):
            result[name] = np.full(count, value, dtype=float)
        return result


class _UniformAttenuation:
    def mu_total(self, energy):
        return 1.0


class RestEnergyTransportTests(unittest.TestCase):
    def _run(self, mode, energy, event):
        """Force one collision per history, followed by real CSDA and banking."""
        state = random.getstate()
        random.seed(1127)
        try:
            data = _UniformAttenuation()
            env = WaterPhantom(data, fixed_energy=energy)
            # Positive radiative values avoid log(0); their contribution is
            # below double precision here. The short charged tracks stop
            # inside the phantom, so their energy must deposit locally.
            stopping = (
                np.array([1e-6, 10.0]),
                np.array([1000.0, 1000.0]),
                np.array([1e-300, 1e-300]),
            )
            runner = (run_learned_transport if mode == "scalar"
                      else run_learned_transport_batched)
            kwargs = {} if mode == "scalar" else {"inference_batch_size": 3}
            with patch("beamweaver.transport.load_stopping_power", return_value=stopping):
                return runner(
                    _FixedCollisionPolicy(event), data, env,
                    n_photons=3, E0=energy, max_steps=1, **kwargs,
                )
        finally:
            random.setstate(state)

    def test_compton_energy_momentum_and_history_accounting(self):
        from beamweaver.constants import PROCESS_INDEX, mec2

        for mode in ("scalar", "batch"):
            for energy, theta, phi in ((0.2, 60.0, 35.0), (3.0, 120.0, 215.0)):
                with self.subTest(mode=mode, energy=energy):
                    event = PhotonCollisionEvent(
                        PROCESS_INDEX["compton"], -1, theta, phi,
                    )
                    tracks, _, dose, records = self._run(mode, energy, event)[:4]
                    self.assertEqual(len(records), 3)
                    expected_out = mec2 / (mec2 / energy + 1.0 - math.cos(math.radians(theta)))
                    scattered_direction = np.array([
                        math.sin(math.radians(theta)) * math.cos(math.radians(phi)),
                        math.sin(math.radians(theta)) * math.sin(math.radians(phi)),
                        math.cos(math.radians(theta)),
                    ])
                    for track, record in zip(tracks, records):
                        self.assertEqual(record["interaction"], "compton")
                        self.assertEqual(record["photon_energy_in"], energy)
                        outgoing = record["photon_energy_out"]
                        self.assertAlmostEqual(outgoing, expected_out, delta=1e-14)
                        species, kinetic, recoil_dir, tag = record["secondaries"][0]
                        self.assertEqual((species, tag), ("electron", "compton_e"))
                        self.assertAlmostEqual(kinetic + outgoing, energy, delta=1e-14)
                        # Use the independent relativistic momentum magnitude,
                        # so a mismatched rest energy cannot hide in E-E_out.
                        recoil_momentum = math.sqrt(kinetic * (kinetic + 2.0 * mec2))
                        np.testing.assert_allclose(
                            np.array([0.0, 0.0, energy]),
                            outgoing * scattered_direction + recoil_momentum * recoil_dir,
                            rtol=0.0, atol=2e-14,
                        )
                        ledger = track["ledger"]
                        deposited = ledger["dep_col"] + ledger["dep_cutoff"] + ledger["dep_soft_rad"]
                        self.assertAlmostEqual(deposited, kinetic, delta=1e-14)
                        self.assertAlmostEqual(ledger["truncated"], outgoing, delta=1e-14)
                        self.assertEqual(ledger["pair_restmass"], 0.0)
                        self.assertEqual(ledger["annih_created"], 0.0)
                        self.assertLessEqual(ledger["closure_err"], 1e-14)
                    self.assertAlmostEqual(float(dose.sum()), 3 * (energy - expected_out), delta=3e-14)

    def test_pair_kinetics_annihilation_and_history_accounting(self):
        from beamweaver.constants import PAIR_THRESHOLD_MEV, PROCESS_INDEX, mec2

        fraction = 0.3
        event = PhotonCollisionEvent(
            PROCESS_INDEX["pair"], -1, 0.0, 0.0,
            pair=(fraction, 20.0, 70.0, 50.0, 190.0),
        )
        # The transport validity check intentionally retains '<': a supplied
        # pair event exactly at threshold is accepted with zero kinetic energy.
        # Neural support masks have their own '<=' boundary, checked elsewhere.
        for mode in ("scalar", "batch"):
            for energy in (PAIR_THRESHOLD_MEV,
                           float(np.nextafter(PAIR_THRESHOLD_MEV, math.inf)), 3.0):
                with self.subTest(mode=mode, energy=energy):
                    tracks, _, dose, records = self._run(mode, energy, event)[:4]
                    self.assertEqual(len(records), 3)
                    available = energy - 2.0 * mec2
                    for track, record in zip(tracks, records):
                        self.assertEqual(record["interaction"], "pair")
                        self.assertEqual(record["photon_energy_out"], 0.0)
                        electron, positron = record["secondaries"]
                        self.assertEqual(electron[0], "electron")
                        self.assertEqual(positron[0], "positron")
                        for secondary, theta, phi in ((electron, 20.0, 70.0),
                                                      (positron, 50.0, 190.0)):
                            expected_direction = [
                                math.sin(math.radians(theta)) * math.cos(math.radians(phi)),
                                math.sin(math.radians(theta)) * math.sin(math.radians(phi)),
                                math.cos(math.radians(theta)),
                            ]
                            np.testing.assert_allclose(secondary[2], expected_direction,
                                                       rtol=0.0, atol=1e-15)
                        self.assertAlmostEqual(electron[1], fraction * available, delta=1e-15)
                        self.assertAlmostEqual(positron[1], (1.0 - fraction) * available, delta=1e-15)
                        self.assertAlmostEqual(electron[1] + positron[1] + 2.0 * mec2,
                                               energy, delta=1e-15)
                        ledger = track["ledger"]
                        self.assertEqual(ledger["pair_restmass"], 2.0 * mec2)
                        self.assertEqual(ledger["annih_created"], 2.0 * mec2)
                        self.assertEqual(ledger["truncated"], 2.0 * mec2)
                        deposited = ledger["dep_col"] + ledger["dep_cutoff"] + ledger["dep_soft_rad"]
                        self.assertAlmostEqual(deposited, available, delta=1e-14)
                        self.assertLessEqual(ledger["closure_err"], 1e-14)
                        bank = track["banked_photons"]
                        self.assertEqual(len(bank), 2)
                        self.assertTrue(all(p["tag"] == "annih" and p["E"] == mec2
                                            and p["fate"] == "truncated" for p in bank))
                        np.testing.assert_array_equal(np.asarray(bank[0]["dir"]),
                                                      -np.asarray(bank[1]["dir"]))
                        self.assertAlmostEqual(float(np.linalg.norm(bank[0]["dir"])),
                                               1.0, delta=1e-14)
                    self.assertAlmostEqual(float(dose.sum()), 3 * available, delta=3e-14)

    def test_both_learned_runners_reject_pairs_below_chosen_threshold(self):
        from beamweaver.constants import PAIR_THRESHOLD_MEV, PROCESS_INDEX

        event = PhotonCollisionEvent(
            PROCESS_INDEX["pair"], -1, 0.0, 0.0,
            pair=(0.3, 20.0, 70.0, 50.0, 190.0),
        )
        old_threshold = 2.0 * 0.5109989  # Historical fixture, never a live default.
        for mode in ("scalar", "batch"):
            for energy in ((old_threshold + PAIR_THRESHOLD_MEV) / 2.0,
                           float(np.nextafter(PAIR_THRESHOLD_MEV, 0.0))):
                with self.subTest(mode=mode, energy=energy):
                    with self.assertRaisesRegex(AssertionError, "pair sampled below threshold"):
                        self._run(mode, energy, event)

    def test_reference_pair_boundary_keeps_existing_endpoint_convention(self):
        from beamweaver.constants import PAIR_THRESHOLD_MEV, mec2

        state = random.getstate()
        random.seed(218)
        direction = np.array([0.0, 0.0, 1.0])
        try:
            before = random.getstate()
            below = float(np.nextafter(PAIR_THRESHOLD_MEV, 0.0))
            rejected = sample_pair_event(below, direction, None)
            self.assertEqual(rejected[3], "pair_subthresh")
            self.assertEqual(rejected[1], below)
            self.assertEqual(rejected[2], [])
            self.assertEqual(random.getstate(), before)
            for energy in (PAIR_THRESHOLD_MEV,
                           float(np.nextafter(PAIR_THRESHOLD_MEV, math.inf)), 3.0):
                with self.subTest(energy=energy):
                    event = sample_pair_event(energy, direction, None)
                    self.assertEqual(event[3], "pair")
                    self.assertEqual(event[1], 0.0)
                    self.assertTrue(all(s[1] >= 0.0 for s in event[2]))
                    self.assertAlmostEqual(sum(s[1] for s in event[2]) + 2.0 * mec2,
                                           energy, delta=1e-15)
                    if energy == PAIR_THRESHOLD_MEV:
                        self.assertEqual([s[1] for s in event[2]], [0.0, 0.0])
        finally:
            random.setstate(state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
