"""Beam Weaver: audit."""

# Copyright (c) 2025–2026 Pedro Teles. Apache-2.0.

from .constants import runtime_physics_metadata
from .constants import (
    PROCESS_INDEX,
    NEGATIVE_ENERGY_ATOL_MEV,
    EVENT_ENERGY_BALANCE_ATOL_MEV,
    KEV_PER_MEV,
)

from contextlib import contextmanager
import csv
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys
from threading import Lock
import time

from .constants import CURRENT_POLICY_SCHEMA, PROCESS_NAMES, SHELL_NAMES



@dataclass
class ExecutionAudit:
    physics_distribution_calls: int = 0
    physics_head_samples: int = 0
    total_events: int = 0
    rayleigh_events: int = 0
    compton_events: int = 0
    photo_events: int = 0
    pair_events: int = 0
    pair_below_threshold: int = 0
    shell_head_samples: int = 0      # learned shell draws
    analytic_shell_calls: int = 0    # blocked reference-shell calls
    invalid_photo_shell: int = 0     # out-of-range shell index
    nonphoto_shell_assignments: int = 0  # shell assigned outside photoelectric events
    kernel_batch_calls: int = 0      # batched sampling invocations
    kernel_sample_rows: int = 0      # sampled event rows
    pair_head_rows: int = 0          # rows through pair heads
    process_head_rows: int = 0
    rayleigh_q2_rows: int = 0
    rayleigh_phi_rows: int = 0
    rayleigh_phi_skipped: int = 0    # axial events (sin t <= eps): no phi
    compton_u_rows: int = 0
    compton_phi_rows: int = 0
    compton_phi_skipped: int = 0
    photo_nu_rows: int = 0
    photo_phi_rows: int = 0
    photo_phi_skipped: int = 0
    pair_fraction_rows: int = 0
    pair_direction_rows: int = 0
    reference_sampler_calls: int = 0 # blocked reference photon-sampler calls
    reference_guard_entries: int = 0
    reference_guard_exits: int = 0
    invalid_theta: int = 0
    invalid_phi: int = 0
    negative_energy: int = 0
    energy_balance_failures: int = 0
    compton_error_max_mev: float = 0.0
    compton_error_sum_mev: float = 0.0
    energy_error_max_mev: float = 0.0
    energy_error_sum_mev: float = 0.0


class TransportExecutionLogger:
    def __init__(self, output_dir, trace_first_n=100, progress_every=1000):
        self.dir = Path(output_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.audit = ExecutionAudit()
        self.trace_first_n = trace_first_n; self.progress_every = progress_every
        self.t0 = time.perf_counter()
        self._tf = (self.dir / "event_trace.csv").open("w", newline="")
        self._tw = csv.DictWriter(self._tf, fieldnames=[
            "event_id", "photon_id", "interaction_index", "source", "process",
            "shell", "E_in_MeV", "free_path_cm", "theta_deg", "phi_deg",
            "E_out_MeV", "secondary_energy_MeV", "x_cm", "y_cm", "z_cm",
            "compton_error_keV", "energy_balance_error_keV"])
        self._tw.writeheader()
        self._pf = (self.dir / "photon_summary.csv").open("w", newline="")
        self._pw = csv.DictWriter(self._pf, fieldnames=[
            "photon_id", "n_interactions", "fate", "final_energy_MeV",
            "deposited_energy_MeV", "escape_energy_MeV"])
        self._pw.writeheader()

    def log_event(self, *, event_id, photon_id, k, process, shell, E_in,
                  free_path, theta_deg, phi_deg, E_out, sec_E, pos,
                  compton_err, ebal_err):
        a = self.audit; a.total_events += 1
        self.log_process_energy(process, E_in)
        attr = f"{PROCESS_NAMES[process]}_events"
        setattr(a, attr, getattr(a, attr) + 1)
        if not 0.0 <= theta_deg <= 180.0: a.invalid_theta += 1
        if not 0.0 <= phi_deg < 360.0: a.invalid_phi += 1
        if process == PROCESS_INDEX["photo"] and not 0 <= shell < len(SHELL_NAMES):
            a.invalid_photo_shell += 1
        if process != PROCESS_INDEX["photo"] and shell >= 0:
            a.nonphoto_shell_assignments += 1
        if E_out < -NEGATIVE_ENERGY_ATOL_MEV or sec_E < -NEGATIVE_ENERGY_ATOL_MEV: a.negative_energy += 1
        a.compton_error_sum_mev += compton_err
        a.compton_error_max_mev = max(a.compton_error_max_mev, compton_err)
        a.energy_error_sum_mev += ebal_err
        a.energy_error_max_mev = max(a.energy_error_max_mev, ebal_err)
        if ebal_err > EVENT_ENERGY_BALANCE_ATOL_MEV: a.energy_balance_failures += 1
        if event_id < self.trace_first_n:
            self._tw.writerow({"event_id": event_id, "photon_id": photon_id,
                "interaction_index": k, "source": "physics_head",
                "process": PROCESS_NAMES[process],
                "shell": SHELL_NAMES[shell] if 0 <= shell < len(SHELL_NAMES) else "",
                "E_in_MeV": E_in, "free_path_cm": free_path,
                "theta_deg": theta_deg, "phi_deg": phi_deg, "E_out_MeV": E_out,
                "secondary_energy_MeV": sec_E, "x_cm": pos[0], "y_cm": pos[1],
                "z_cm": pos[2], "compton_error_keV": KEV_PER_MEV * compton_err,
                "energy_balance_error_keV": KEV_PER_MEV * ebal_err})
            self._tf.flush()
        if a.total_events % self.progress_every == 0:
            self.print_progress()

    def log_photon(self, photon_id, n_int, fate, E_final, E_dep, E_esc):
        self._pw.writerow({"photon_id": photon_id, "n_interactions": n_int,
                           "fate": fate, "final_energy_MeV": E_final,
                           "deposited_energy_MeV": E_dep,
                           "escape_energy_MeV": E_esc})

    def print_progress(self):
        a = self.audit; n = max(a.total_events, 1)
        rate = a.total_events / max(time.perf_counter() - self.t0, 1e-12)
        print(f"[physics-head inference] events={a.total_events:,} "
              f"rate={rate:,.0f}/s | R={a.rayleigh_events/n:.3f} "
              f"C={a.compton_events/n:.3f} P={a.photo_events/n:.3f} | "
              f"max dE_C={KEV_PER_MEV*a.compton_error_max_mev:.2e} keV "
              f"max dE={KEV_PER_MEV*a.energy_error_max_mev:.2e} keV")

    def log_process_energy(self, process, E_in):
        if not hasattr(self, "_pe_counts"):
            self._pe_counts = {}
        key = f"{PROCESS_NAMES[process]}_{'%.0f' % (E_in * KEV_PER_MEV)}keV" \
            if E_in < 0.01 else \
            f"{PROCESS_NAMES[process]}_{int(E_in * KEV_PER_MEV)//50*50}-{int(E_in*KEV_PER_MEV)//50*50+50}keV"
        self._pe_counts[key] = self._pe_counts.get(key, 0) + 1

    def finalize(self):
        self._tf.close(); self._pf.close()
        a = self.audit
        # The caller supplies these descriptions. Counter checks below are
        # operational evidence; metadata alone cannot establish provenance.
        declared = {
            "kernel_architecture": getattr(self, "kernel_architecture", None),
            "checkpoint_schema": getattr(self, "checkpoint_schema", None),
            "free_path_source": getattr(self, "free_path_source", None),
            "shell_mode": getattr(self, "shell_mode", None),
        }
        checks = {
            "current_policy_declared": (
                declared["kernel_architecture"] == "GenerativeTransportPolicy"
                and declared["checkpoint_schema"] == CURRENT_POLICY_SCHEMA
                and declared["free_path_source"] == "analytic_tabulated"
                and declared["shell_mode"] == "learned"),
            "event_rows_accounted": (
                a.physics_head_samples == a.total_events
                and a.physics_distribution_calls == a.total_events
                and a.process_head_rows == a.total_events),
            "conditional_rows_accounted": (
                a.shell_head_samples == a.photo_events
                and a.rayleigh_q2_rows == a.rayleigh_events
                and a.compton_u_rows == a.compton_events
                and a.photo_nu_rows == a.photo_events
                and a.pair_head_rows == a.pair_events
                and a.pair_fraction_rows == a.pair_events
                and a.pair_direction_rows == a.pair_events),
            "azimuth_rows_accounted": (
                a.rayleigh_phi_rows + a.rayleigh_phi_skipped == a.rayleigh_events
                and a.compton_phi_rows + a.compton_phi_skipped == a.compton_events
                and a.photo_phi_rows + a.photo_phi_skipped == a.photo_events),
            "reference_guard_completed": (
                a.reference_guard_entries > 0
                and a.reference_guard_entries == a.reference_guard_exits),
            "no_reference_sampler_calls": (
                a.reference_sampler_calls == 0 and a.analytic_shell_calls == 0),
            "event_diagnostics_pass": (
                a.energy_balance_failures == 0 and a.negative_energy == 0
                and a.invalid_theta == 0 and a.invalid_phi == 0
                and a.pair_below_threshold == 0 and a.invalid_photo_shell == 0
                and a.nonphoto_shell_assignments == 0),
        }
        verified = all(checks.values())
        summary = {
            **asdict(a),
            "runtime_physics_constants": runtime_physics_metadata(),
            "declared_configuration": declared,
            "audit_checks": checks,
            "diagnostic_sources": {
                "event_diagnostics": "Computed from logged event values; energy errors are supplied by transport.",
                "head_rows": "Counted by policy sampling code and compared with logged events.",
                "reference_calls": "Counted by the process-wide reference-sampler guard.",
                "configuration": "Declared by the evaluation caller, not independently measured.",
            },
            "elapsed_seconds": time.perf_counter() - self.t0,
            "mean_compton_error_keV": KEV_PER_MEV * a.compton_error_sum_mev / max(a.compton_events, 1),
            "mean_energy_balance_error_keV": KEV_PER_MEV * a.energy_error_sum_mev / max(a.total_events, 1),
            "process_energy_counts": getattr(self, "_pe_counts", {}),
            "shell_mode": declared["shell_mode"],
            "inference_source_verified": verified,
        }
        with (self.dir / "execution_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        if not verified:
            raise RuntimeError("inference-source audit failed; see execution_summary.json")
        return summary


_REFERENCE_GUARD_LOCK = Lock()


@contextmanager
def reference_sampler_guard(audit):
    """Block reference photon sampling while the learned arm transports.

    This process-wide guard patches the defining functions, their aliases
    in loaded beamweaver modules, and reference sampling class methods.
    It permits shared lepton transport and material-table lookup. Do not
    run reference MC or dataset generation concurrently in this process.
    A retained callable outside the package is outside this guard's scope.
    """
    if not _REFERENCE_GUARD_LOCK.acquire(blocking=False):
        raise RuntimeError("a reference-sampler guard is already active in this process")

    saved = []
    replacements = {}
    package = __package__.split(".")[0]
    entered = False
    try:
        from . import materials, physics

        functions = [
            (name, getattr(physics, name), False)
            for name in ("sample_rayleigh_event", "sample_compton_event", "sample_photoelectric_event",
                         "sample_photoelectric_event_for_shell", "sample_pair_event", "sample_photon_interaction")
        ]
        methods = [
            (physics.ComptonEnergyTransferSampler, "sample_energy_transfer", False),
            (materials.WaterPhotoShellData, "sample_shell", True),
            (materials.WaterPhotonData, "sample_photo_shell_index", True),
        ]
        for owner, name, is_shell in methods:
            functions.append((f"{owner.__name__}.{name}", getattr(owner, name), is_shell))

        replacements = {}
        for label, original, is_shell in functions:
            def blocked(*args, _label=label, _shell=is_shell, **kwargs):
                if audit is not None:
                    audit.reference_sampler_calls += 1
                    if _shell:
                        audit.analytic_shell_calls += 1
                raise RuntimeError(f"reference sampler '{_label}' invoked inside learned transport")
            replacements[id(original)] = (original, blocked)

        package = __package__.split(".")[0]
        for module_name, module in list(sys.modules.items()):
            if module is None or not (module_name == package or module_name.startswith(package + ".")):
                continue
            for name, value in list(vars(module).items()):
                replacement = replacements.get(id(value))
                if replacement is not None and value is replacement[0]:
                    saved.append((module, name, value))
                    setattr(module, name, replacement[1])
        for owner, name, _ in methods:
            original = getattr(owner, name)
            saved.append((owner, name, original))
            setattr(owner, name, replacements[id(original)][1])
        if audit is not None:
            audit.reference_guard_entries += 1
        entered = True
        yield
    finally:
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)
        # Modules imported during the guard may have captured a tripwire
        # through a from-import. Restore those newly created aliases too.
        originals = {id(blocked): (blocked, original)
                     for original, blocked in replacements.values()}
        for module_name, module in list(sys.modules.items()):
            if module is None or not (module_name == package or module_name.startswith(package + ".")):
                continue
            for name, value in list(vars(module).items()):
                original = originals.get(id(value))
                if original is not None and value is original[0]:
                    setattr(module, name, original[1])
        if entered and audit is not None:
            audit.reference_guard_exits += 1
        _REFERENCE_GUARD_LOCK.release()
