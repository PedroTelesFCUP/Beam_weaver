"""Pair support at the chosen rest energy using actual CPU Torch policies.

Artificial logits make the selected process deterministic; these checks do
not claim accuracy of a trained collision distribution. Existing log32
transport encoding remains part of the deployed input route.
"""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if not (PROJECT / "beamweaver").is_dir():
    PROJECT = PROJECT / "BeamWeaver"
sys.path.insert(0, str(PROJECT))
HAS_TORCH = importlib.util.find_spec("torch") is not None

# Historical numerical value is deliberately retained only as a test input.
OLD_PAIR_THRESHOLD_MEV = 2.0 * 0.5109989


def _edges_fixture():
    return {
        "ray_q2": np.array([-12.0, -3.0, 0.0]),
        "comp_u": np.array([0.0, 0.5, 1.0]),
        "photo_nu": np.array([0.0, 1.0, 2.0]),
        "phi": np.array([0.0, np.pi, 2.0 * np.pi]),
        "pair_frac": np.array([0.0, 0.5, 1.0]),
        "pair_nu_e": np.array([0.0, 1.0, 2.0]),
        "pair_nu_p": np.array([0.0, 1.0, 2.0]),
    }


@unittest.skipUnless(HAS_TORCH, "Torch unavailable: policy boundary checks not executed")
class RestEnergyPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from beamweaver.constants import (
            BW4_EMAX_MEV, BW4_EMIN_MEV, PAIR_THRESHOLD_MEV, PROCESS_INDEX, mec2,
        )
        from beamweaver.policy import GenerativeTransportPolicy

        cls.torch = torch
        cls.threshold = PAIR_THRESHOLD_MEV
        if cls.threshold != 2.0 * mec2:
            raise AssertionError("Pair support must derive from the canonical mec2")
        cls.pair_id = PROCESS_INDEX["pair"]
        cls.ray_id = PROCESS_INDEX["rayleigh"]
        cls.env = SimpleNamespace(E_min=BW4_EMIN_MEV, E_max=BW4_EMAX_MEV)
        cls.model = GenerativeTransportPolicy(_edges_fixture())
        with torch.no_grad():
            for parameter in cls.model.parameters():
                parameter.zero_()
            # exp(-2000) underflows even in double precision: pair above
            # threshold, Rayleigh when the pair logit is masked.
            bias = cls.model.process_head.net[-1].bias
            bias.fill_(-2000.0)
            bias[cls.ray_id] = 0.0
            bias[cls.pair_id] = 2000.0
        cls.model.ready = {factor: True for factor in cls.model.ready}
        cls.model.eval()
        cls.pair_head_names = (
            "pair_fraction_head", "pair_electron_polar_head",
            "pair_electron_phi_head", "pair_positron_polar_head",
            "pair_positron_phi_head",
        )

    def setUp(self):
        self.head_calls = {name: 0 for name in self.pair_head_names}
        self.hooks = []
        for name in self.pair_head_names:
            def counted(_module, _inputs, _output, name=name):
                self.head_calls[name] += 1
            self.hooks.append(getattr(self.model, name).register_forward_hook(counted))

    def tearDown(self):
        for hook in self.hooks:
            hook.remove()

    def _generator(self):
        return self.torch.Generator(device="cpu").manual_seed(17019)

    def _clear_calls(self):
        for name in self.head_calls:
            self.head_calls[name] = 0

    def _assert_scalar_support(self, energy):
        self._clear_calls()
        energy = float(energy)
        expected_pair = energy > self.threshold
        event = self.model.generate_event(energy, generator=self._generator())
        self.assertEqual(event.process, self.pair_id if expected_pair else self.ray_id)
        self.assertEqual(event.pair is not None, expected_pair)
        if expected_pair:
            self.assertTrue(np.all(np.isfinite(event.pair)))
        self.assertEqual(self.head_calls, {
            name: int(expected_pair) for name in self.pair_head_names
        })

    def _assert_batch_support(self, energies):
        self._clear_calls()
        energies = np.asarray(energies, dtype=np.float64)
        expected_pair = energies > self.threshold
        result = self.model.generate_events_batch(energies, generator=self._generator())
        np.testing.assert_array_equal(result["process"],
                                      np.where(expected_pair, self.pair_id, self.ray_id))
        for field in ("pair_frac", "pair_e_theta", "pair_e_phi",
                      "pair_p_theta", "pair_p_phi"):
            self.assertTrue(np.all(np.isnan(result[field][~expected_pair])), field)
            self.assertTrue(np.all(np.isfinite(result[field][expected_pair])), field)
        self.assertEqual(self.head_calls, {
            name: int(expected_pair.any()) for name in self.pair_head_names
        })

    def _boundary_inputs(self):
        t = self.threshold
        tf = np.float32(t)
        return np.array([
            np.nextafter(t, -np.inf), t, np.nextafter(t, np.inf),
            float(np.nextafter(tf, np.float32(-np.inf))), float(tf),
            float(np.nextafter(tf, np.float32(np.inf))),
            OLD_PAIR_THRESHOLD_MEV,
            0.5 * (OLD_PAIR_THRESHOLD_MEV + t),
        ], dtype=np.float64)

    def _encoding_inputs(self):
        # Include adjacent representable log32 values so the actual encoding
        # routes exercise both sides even where raw nextafter energies collapse.
        log_t = np.float32(np.log10(self.threshold))
        logs = np.array([
            np.nextafter(log_t, np.float32(-np.inf)), log_t,
            np.nextafter(log_t, np.float32(np.inf)),
        ], dtype=np.float32)
        return np.r_[self._boundary_inputs(), 10.0 ** logs.astype(np.float64)]

    def test_scalar_physical_support_at_float64_and_float32_boundaries(self):
        for energy in self._boundary_inputs():
            with self.subTest(energy=repr(float(energy))):
                self._assert_scalar_support(energy)

    def test_batch_physical_support_at_float64_and_float32_boundaries(self):
        self._assert_batch_support(self._boundary_inputs())
        # An entirely forbidden wave must not call any pair head.
        self._assert_batch_support([np.nextafter(self.threshold, -np.inf),
                                    self.threshold])

    def test_equal_float32_features_do_not_collapse_physical_support(self):
        low = np.nextafter(self.threshold, -np.inf)
        high = np.nextafter(self.threshold, np.inf)
        x_low = self.model._energy_features(low)
        x_high = self.model._energy_features(high)
        self.assertEqual(x_low.dtype, self.torch.float32)
        self.assertTrue(self.torch.equal(x_low, x_high))
        self._assert_scalar_support(low)
        self._assert_scalar_support(high)
        self._assert_batch_support([low, high])

    def test_old_threshold_interval_is_now_forbidden(self):
        energies = np.array([
            np.nextafter(OLD_PAIR_THRESHOLD_MEV, np.inf),
            0.5 * (OLD_PAIR_THRESHOLD_MEV + self.threshold),
            np.nextafter(self.threshold, -np.inf), self.threshold,
        ])
        self.assertTrue(np.all(energies > OLD_PAIR_THRESHOLD_MEV))
        self.assertTrue(np.all(energies <= self.threshold))
        for energy in energies:
            self._assert_scalar_support(energy)
        self._assert_batch_support(energies)

    def test_scalar_transport_encoding_route(self):
        from beamweaver.geometry import encode_policy_energy

        encoded = np.array([encode_policy_energy(float(e), self.env)
                            for e in self._encoding_inputs()])
        self.assertTrue(np.any(encoded < self.threshold))
        self.assertTrue(np.any(encoded > self.threshold))
        for energy in encoded:
            with self.subTest(encoded_energy=repr(float(energy))):
                self._assert_scalar_support(energy)

    def test_batch_transport_encoding_route(self):
        from beamweaver.geometry import encode_policy_energy_batch

        encoded = encode_policy_energy_batch(self._encoding_inputs(), self.env)
        self.assertEqual(encoded.dtype, np.float64)
        self.assertTrue(np.any(encoded < self.threshold))
        self.assertTrue(np.any(encoded > self.threshold))
        self._assert_batch_support(encoded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
