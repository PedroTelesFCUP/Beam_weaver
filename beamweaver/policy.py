"""Factorized generative photon-collision policy."""

# Copyright (c) 2025–2026 Pedro Teles & João Melo. Apache-2.0.

import hashlib
import math

import numpy as np
import torch
import torch.nn as nn

from .constants import runtime_physics_metadata
from .constants import PAIR_THRESHOLD_MEV
from .constants import (
    BW4_COORDS,
    BW4_FACTOR_HEADS,
    BW4_NORM_LOGE_SPEC,
    BW4_PHI_EPS,
    BW_VERSION,
    CURRENT_POLICY_SCHEMA,
    N_PROCESSES,
    N_SHELLS,
    PROCESS_INDEX,
    PROCESS_NAMES,
    SHELL_NAMES,
)
from .coordinates import (
    cos_from_lq2r,
    cos_from_tau,
    norm_logE,
    tau_from_u,
)
from .events import PhotonCollisionEvent


class ConditionalCategoricalHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnconditionalCategoricalHead(nn.Module):
    """Trainable categorical logits for factors with no conditioning input."""
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x=None):
        if x is None or (torch.is_tensor(x) and x.dim() == 0):
            return self.logits.unsqueeze(0)
        n = x.shape[0] if torch.is_tensor(x) else int(x)
        return self.logits.unsqueeze(0).expand(n, -1)


class GenerativeTransportPolicy(nn.Module):
    """A factorized photon-collision policy π(Z|E), with energy in MeV.

    Heads have disjoint parameters. Event generation samples categorical bins
    and within-bin coordinates, then applies the physical inverse transforms.
    It does not call the reference collision samplers.
    """

    def __init__(self, edges: dict):
        super().__init__()
        self.edge_arrays = {k: np.asarray(v, np.float64)
                            for k, v in edges.items()}
        for k, v in self.edge_arrays.items():
            self.register_buffer(f"edges_{k}",
                                 torch.tensor(v, dtype=torch.float64))
        K = {k: len(v) - 1 for k, v in self.edge_arrays.items()}
        outs = {"ray_q2": K["ray_q2"], "ray_phi": K["phi"],
                "comp_u": K["comp_u"], "comp_phi": K["phi"],
                "photo_nu": K["photo_nu"], "photo_phi": K["phi"],
                "pair_frac": K["pair_frac"],
                "pair_e_polar": K["pair_nu_e"], "pair_e_phi": K["phi"],
                "pair_p_polar": K["pair_nu_p"], "pair_p_phi": K["phi"]}
        for fac, (attr, ind, outd) in BW4_FACTOR_HEADS.items():
            outd = outs.get(fac, outd)
            torch.manual_seed(int.from_bytes(
                hashlib.sha256(f"v040-init|{fac}".encode()).digest()[:4],
                "big"))
            head = (UnconditionalCategoricalHead(outd) if ind == 0
                    else ConditionalCategoricalHead(ind, outd))
            setattr(self, attr, head)
        self.ready = {f: False for f in BW4_FACTOR_HEADS}
        self.n_interactions = N_PROCESSES
        self.reference_sampler_calls = 0
        self.meta = {}

    def _dev(self):
        return next(self.parameters()).device

    def _require(self, *facs):
        missing = [f for f in facs if not self.ready.get(f)]
        if missing:
            raise RuntimeError(
                f"Factors not trained: {missing}; train them before inference")

    @staticmethod
    def _draw_cat(logits, gen):
        p = torch.softmax(logits.double(), dim=-1)
        return int(torch.multinomial(p, 1, generator=gen).item())

    def _uni(self, gen):
        return float(torch.rand(1, generator=gen,
                                device=self._dev()).item())

    def _sample_bin_value(self, name, k, gen):
        e = self.edge_arrays[name if name in self.edge_arrays else "phi"]
        lo, hi = float(e[k]), float(e[k + 1])
        return lo + self._uni(gen) * (hi - lo)

    def _energy_features(self, E):
        return torch.tensor([[float(norm_logE(E))]],
                            dtype=torch.float32, device=self._dev())

    def generate_event(self, E, generator=None, audit=None):
        """Sample one collision event at photon energy ``E`` in MeV."""
        gen = generator
        self._require("process")
        xE = self._energy_features(E)
        pl = self.process_head(xE)[0].clone()
        if E <= PAIR_THRESHOLD_MEV:      # boundary itself has zero phase space
            pl[PROCESS_INDEX["pair"]] = float("-inf")
        D = self._draw_cat(pl, gen)
        if audit is not None:
            # The event logger owns total and per-process event tallies.
            audit.physics_distribution_calls += 1
            audit.physics_head_samples += 1
            audit.process_head_rows += 1
        shell = -1
        theta_deg = phi_deg = 0.0
        pair = None
        if D == PROCESS_INDEX["rayleigh"]:            # Rayleigh — s, φ
            self._require("ray_q2", "ray_phi")
            k = self._draw_cat(self.rayleigh_q2_head(xE)[0], gen)
            sv = self._sample_bin_value("ray_q2", k, gen)   # s=ln(q2/q2max): no mask
            theta = math.acos(max(-1.0, min(1.0,
                                            float(cos_from_lq2r(sv)))))
            phi = self._sample_bin_value("phi",
                               self._draw_cat(self.rayleigh_phi_head(), gen),
                               gen)
            theta_deg, phi_deg = math.degrees(theta), math.degrees(phi)
            if audit is not None:
                audit.rayleigh_q2_rows += 1
                if math.sin(theta) > BW4_PHI_EPS:
                    audit.rayleigh_phi_rows += 1
                else:
                    audit.rayleigh_phi_skipped += 1
        elif D == PROCESS_INDEX["compton"]:           # Compton — u, φ
            self._require("comp_u", "comp_phi")
            k = self._draw_cat(self.compton_u_head(xE)[0], gen)
            u = self._sample_bin_value("comp_u", k, gen)
            tau = float(tau_from_u(u, E))
            theta = math.acos(max(-1.0, min(1.0, cos_from_tau(tau, E))))
            phi = self._sample_bin_value("phi",
                               self._draw_cat(self.compton_phi_head(), gen),
                               gen)
            theta_deg, phi_deg = math.degrees(theta), math.degrees(phi)
            if audit is not None:
                audit.compton_u_rows += 1
                if math.sin(theta) > BW4_PHI_EPS:
                    audit.compton_phi_rows += 1
                else:
                    audit.compton_phi_skipped += 1
        elif D == PROCESS_INDEX["photo"]:             # photo — H, ν, φ
            self._require("shell", "photo_nu", "photo_phi")
            shell = self._draw_cat(self.shell_head(xE)[0], gen)
            soh = torch.zeros(1, N_SHELLS, device=self._dev()); soh[0, shell] = 1.0
            k = self._draw_cat(self.photo_nu_head(
                torch.cat([xE, soh], dim=1))[0], gen)
            nu = self._sample_bin_value("photo_nu", k, gen)
            theta = math.acos(max(-1.0, min(1.0, 1.0 - nu)))
            phi = self._sample_bin_value("phi",
                               self._draw_cat(self.photo_phi_head(soh)[0],
                                              gen), gen)
            theta_deg, phi_deg = math.degrees(theta), math.degrees(phi)
            if audit is not None:
                audit.shell_head_samples += 1
                audit.photo_nu_rows += 1
                if math.sin(theta) > BW4_PHI_EPS:
                    audit.photo_phi_rows += 1
                else:
                    audit.photo_phi_skipped += 1
        else:                                        # pair — f, Ω−, Ω+
            self._require("pair_frac", "pair_e_polar", "pair_e_phi",
                          "pair_p_polar", "pair_p_phi")
            k = self._draw_cat(self.pair_fraction_head(xE)[0], gen)
            f = self._sample_bin_value("pair_frac", k, gen)
            xe = torch.tensor([[float(norm_logE(E)), f]],
                             dtype=torch.float32, device=self._dev())
            xp = torch.tensor([[float(norm_logE(E)), 1.0 - f]],
                             dtype=torch.float32, device=self._dev())
            ke = self._draw_cat(self.pair_electron_polar_head(xe)[0], gen)
            kp = self._draw_cat(self.pair_positron_polar_head(xp)[0], gen)
            nu_e = self._sample_bin_value("pair_nu_e", ke, gen)
            nu_p = self._sample_bin_value("pair_nu_p", kp, gen)
            th_e = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - nu_e))))
            th_p = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - nu_p))))
            ph_e = math.degrees(self._sample_bin_value(
                "phi", self._draw_cat(self.pair_electron_phi_head(xe)[0],
                                      gen), gen))
            ph_p = math.degrees(self._sample_bin_value(
                "phi", self._draw_cat(self.pair_positron_phi_head(xp)[0],
                                      gen), gen))
            pair = (f, th_e, ph_e, th_p, ph_p)
            if audit is not None:
                audit.pair_fraction_rows += 1
                audit.pair_direction_rows += 1
                audit.pair_head_rows += 1
        return PhotonCollisionEvent(process=D, shell=shell, theta_deg=theta_deg,
                            phi_deg=phi_deg, pair=pair)

    def generate_events_batch(self, energies, generator=None, audit=None):
        """Sample collision events from a 1D array of photon energies in MeV.

        Run one process-head pass over the wave and one pass per conditional
        head over the relevant subset. Return arrays keyed by event field.
        """
        dev = self._dev()
        self._require("process")
        E = np.asarray(energies, dtype=np.float64)
        if E.ndim != 1:
            raise ValueError("energies must be a one-dimensional array in MeV")
        B = len(E)
        thr = PAIR_THRESHOLD_MEV
        xE = torch.tensor(norm_logE(E), dtype=torch.float32,
                          device=dev).unsqueeze(1)

        def _mnom(logits):
            pr = torch.softmax(logits.double(), -1)
            return torch.multinomial(pr, 1,
                                     generator=generator).squeeze(1)

        def _u(n):
            return torch.rand(n, generator=generator, device=dev,
                              dtype=torch.float64).cpu().numpy()

        Kphi = len(self.edge_arrays["phi"]) - 1
        dphi = 2.0 * math.pi / Kphi
        out = {k: np.full(B, np.nan) for k in
               ("theta_deg", "phi_deg", "pair_frac", "pair_e_theta",
                "pair_e_phi", "pair_p_theta", "pair_p_phi")}
        with torch.no_grad():
            pl = self.process_head(xE)
            pl = pl.masked_fill(torch.tensor(E <= thr, device=dev)
                                .unsqueeze(1)
                                & (torch.arange(N_PROCESSES, device=dev)
                                   == PROCESS_INDEX["pair"]),
                                float("-inf"))
            Dn = _mnom(pl).cpu().numpy()
            out["process"] = Dn.astype(np.int64)
            out["shell"] = np.full(B, -1, np.int64)
            if audit is not None:
                audit.physics_distribution_calls += B
                audit.physics_head_samples += B
                audit.process_head_rows += B
                audit.kernel_batch_calls += 1
                audit.kernel_sample_rows += B

            def _phi_counts(sin_ok, rows_attr, skip_attr):
                if audit is not None:
                    setattr(audit, rows_attr,
                            getattr(audit, rows_attr) + int(sin_ok.sum()))
                    setattr(audit, skip_attr,
                            getattr(audit, skip_attr)
                            + int((~sin_ok).sum()))

            idx = np.nonzero(Dn == PROCESS_INDEX["rayleigh"])[0]
            if len(idx):
                self._require("ray_q2", "ray_phi")
                e = self.edge_arrays["ray_q2"]; a, b = e[:-1], e[1:]
                k = _mnom(self.rayleigh_q2_head(xE[idx])).cpu().numpy()
                sv = a[k] + _u(len(idx)) * (b[k] - a[k])
                cth = np.clip(cos_from_lq2r(sv), -1.0, 1.0)
                out["theta_deg"][idx] = np.degrees(np.arccos(cth))
                kph = _mnom(self.rayleigh_phi_head(len(idx))).cpu().numpy()
                out["phi_deg"][idx] = np.degrees((kph + _u(len(idx)))
                                                 * dphi)
                if audit is not None:
                    audit.rayleigh_q2_rows += len(idx)
                _phi_counts(np.sqrt(1 - cth**2) > BW4_PHI_EPS,
                            "rayleigh_phi_rows", "rayleigh_phi_skipped")

            idx = np.nonzero(Dn == PROCESS_INDEX["compton"])[0]
            if len(idx):
                self._require("comp_u", "comp_phi")
                e = self.edge_arrays["comp_u"]; a, b = e[:-1], e[1:]
                k = _mnom(self.compton_u_head(xE[idx])).cpu().numpy()
                u = a[k] + _u(len(idx)) * (b[k] - a[k])
                tau = tau_from_u(u, E[idx])
                cth = np.clip(cos_from_tau(tau, E[idx]), -1.0, 1.0)
                out["theta_deg"][idx] = np.degrees(np.arccos(cth))
                kph = _mnom(self.compton_phi_head(len(idx))).cpu().numpy()
                out["phi_deg"][idx] = np.degrees((kph + _u(len(idx)))
                                                 * dphi)
                if audit is not None:
                    audit.compton_u_rows += len(idx)
                _phi_counts(np.sqrt(1 - cth**2) > BW4_PHI_EPS,
                            "compton_phi_rows", "compton_phi_skipped")

            idx = np.nonzero(Dn == PROCESS_INDEX["photo"])[0]
            if len(idx):
                self._require("shell", "photo_nu", "photo_phi")
                sh = _mnom(self.shell_head(xE[idx]))
                soh = torch.nn.functional.one_hot(sh, N_SHELLS).float()
                out["shell"][idx] = sh.cpu().numpy()
                e = self.edge_arrays["photo_nu"]; a, b = e[:-1], e[1:]
                k = _mnom(self.photo_nu_head(
                    torch.cat([xE[idx], soh], dim=1))).cpu().numpy()
                nu = a[k] + _u(len(idx)) * (b[k] - a[k])
                cth = np.clip(1.0 - nu, -1.0, 1.0)
                out["theta_deg"][idx] = np.degrees(np.arccos(cth))
                kph = _mnom(self.photo_phi_head(soh)).cpu().numpy()
                out["phi_deg"][idx] = np.degrees((kph + _u(len(idx)))
                                                 * dphi)
                if audit is not None:
                    audit.shell_head_samples += len(idx)
                    audit.photo_nu_rows += len(idx)
                _phi_counts(np.sqrt(1 - cth**2) > BW4_PHI_EPS,
                            "photo_phi_rows", "photo_phi_skipped")

            idx = np.nonzero(Dn == PROCESS_INDEX["pair"])[0]
            if len(idx):
                self._require("pair_frac", "pair_e_polar", "pair_e_phi",
                              "pair_p_polar", "pair_p_phi")
                e = self.edge_arrays["pair_frac"]; a, b = e[:-1], e[1:]
                k = _mnom(self.pair_fraction_head(xE[idx])).cpu().numpy()
                f = a[k] + _u(len(idx)) * (b[k] - a[k])
                ft = torch.tensor(f, dtype=torch.float32,
                                  device=dev).unsqueeze(1)
                xe = torch.cat([xE[idx], ft], dim=1)
                xp = torch.cat([xE[idx], 1.0 - ft], dim=1)
                for lep, xin in (("e", xe), ("p", xp)):
                    ee = self.edge_arrays[f"pair_nu_{lep}"]
                    aa, bb = ee[:-1], ee[1:]
                    hd = (self.pair_electron_polar_head if lep == "e"
                          else self.pair_positron_polar_head)
                    kk = _mnom(hd(xin)).cpu().numpy()
                    nu = aa[kk] + _u(len(idx)) * (bb[kk] - aa[kk])
                    out[f"pair_{lep}_theta"][idx] = np.degrees(
                        np.arccos(np.clip(1.0 - nu, -1.0, 1.0)))
                    hp = (self.pair_electron_phi_head if lep == "e"
                          else self.pair_positron_phi_head)
                    kp = _mnom(hp(xin)).cpu().numpy()
                    out[f"pair_{lep}_phi"][idx] = np.degrees(
                        (kp + _u(len(idx))) * dphi)
                out["pair_frac"][idx] = f
                out["theta_deg"][idx] = 0.0
                out["phi_deg"][idx] = 0.0
                if audit is not None:
                    audit.pair_fraction_rows += len(idx)
                    audit.pair_direction_rows += len(idx)
                    audit.pair_head_rows += len(idx)
        return out

    @property
    def pair_heads_ready(self):
        return all(self.ready[f] for f in
                   ("pair_frac", "pair_e_polar", "pair_e_phi",
                    "pair_p_polar", "pair_p_phi"))

    def save_policy(self, path, extra=None):
        ck = {"policy_schema": CURRENT_POLICY_SCHEMA,
              "runtime_physics_constants": runtime_physics_metadata(),
              "architecture": "GenerativeTransportPolicy",
              "state_dict": self.state_dict(),
              "edges": {k: v for k, v in self.edge_arrays.items()},
              "edge_hashes": {k: hashlib.sha256(v.tobytes()).hexdigest()
                              for k, v in self.edge_arrays.items()},
              "ready": dict(self.ready),
              "norm_logE": BW4_NORM_LOGE_SPEC,
              "coordinates": BW4_COORDS,
              "process_order": PROCESS_NAMES, "shell_order": SHELL_NAMES,
              "meta": {**self.meta, **(extra or {})}}
        torch.save(ck, path)
        return ck

    @classmethod
    def load_policy(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        assert ck.get("policy_schema") == CURRENT_POLICY_SCHEMA, \
            f"not a v{BW_VERSION} policy"
        pol = cls(ck["edges"])
        pol.load_state_dict(ck["state_dict"])
        pol.ready = dict(ck["ready"])
        pol.meta = dict(ck.get("meta", {}))
        pol.to(device)
        pol.eval()
        return pol
