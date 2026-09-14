"""Beam Weaver: transport."""

# Copyright (c) 2025–2026 Pedro Teles. Apache-2.0.

from .constants import PAIR_THRESHOLD_MEV
from .constants import (
    DEFAULT_TRANSPORT_ECUT_MEV,
    DEFAULT_INFERENCE_BATCH_SIZE,
    SOURCE_FIELD_WIDTH_CM,
    SOURCE_FIELD_HALF_WIDTH_CM,
    ATTENUATION_FLOOR_CM_INV,
    FREE_PATH_RANDOM_FLOOR,
    LEARNED_RECOIL_MOMENTUM_FLOOR_MEV_C,
    HISTORY_CLOSURE_ATOL_MEV,
    ELECTRON_STOPPING_POWER_TABLE,
    PERFORMANCE_TIME_FLOOR_S,
)

from .constants import TRANSPORT_MAX_STEPS
from .constants import mec2
from .constants import PHOTO_SHELL_BINDINGS_MEV
from .constants import PROCESS_NAMES
from .materials import WaterPhotonData
from .constants import SHELL_NAMES
from .geometry import encode_policy_energy, encode_policy_energy_batch
from .physics import load_stopping_power
import math
import numpy as np
from .physics import sample_photon_interaction
import random
from .physics import rotate_direction
import time
from .physics import transport_lepton_csda


def run_reference_transport(n_photons, data: WaterPhotonData, env, max_steps=TRANSPORT_MAX_STEPS,
                  ecut=DEFAULT_TRANSPORT_ECUT_MEV, verbose=False):
    """Reference Monte Carlo transport with a photon bank for each history.

    The closed energy ledger records deposits, escaped and discarded
    energy, pair rest mass, and annihilation energy. Radiated photons are
    banked with their parent, generation, birth position, and terminal fate.
    NaN coordinate separators keep unrelated photon paths distinct.

    Closure, in MeV:
        E_primary + annih_created = deposits + gamma_escaped
            + charged_escaped + subcutoff_dropped + binding_dropped
            + pair_restmass + truncated
    """
    tracks = []
    all_secondaries = []
    mc_interactions = []
    stopping_Egrid, stopping_S_col, stopping_S_rad = load_stopping_power(ELECTRON_STOPPING_POWER_TABLE)
    env.reset()
    env.dose_tally[:] = 0.0

    for hist in range(n_photons):
        px0 = (random.random() * SOURCE_FIELD_WIDTH_CM) - SOURCE_FIELD_HALF_WIDTH_CM
        py0 = (random.random() * SOURCE_FIELD_WIDTH_CM) - SOURCE_FIELD_HALF_WIDTH_CM
        # bank entry: (E, dir, pos, tag, generation, record-or-None)
        photon_bank = [(env.fixed_energy, (0.0, 0.0, 1.0),
                        (px0, py0, 0.0), "primary", 0, None)]
        single_x, single_y, single_z = [], [], []
        photon_interactions = []
        secondaries = []
        banked_records = []
        led = {"E_in": float(env.fixed_energy), "dep_col": 0.0, "dep_cutoff": 0.0,
               "dep_soft_rad": 0.0, "gamma_escaped": 0.0, "charged_escaped": 0.0,
               "subcutoff_dropped": 0.0, "binding_dropped": 0.0,
               "pair_restmass": 0.0, "annih_created": 0.0, "truncated": 0.0}
        steps = 0
        first_segment = True

        while photon_bank and steps < max_steps:
            Eg, dg, pg, tag, gen, rec = photon_bank.pop()
            if Eg <= ecut:
                led["subcutoff_dropped"] += Eg
                if rec is not None:
                    rec["fate"] = "subcutoff"
                continue
            if not (env.xmin <= pg[0] <= env.xmax and
                    env.ymin <= pg[1] <= env.ymax and
                    env.zmin <= pg[2] <= env.zmax):
                led["gamma_escaped"] += Eg          # born outside → escapes
                if rec is not None:
                    rec["fate"] = "born_outside"
                continue
            if not first_segment:                    # NaN track separator
                single_x.append(float("nan"))
                single_y.append(float("nan"))
                single_z.append(float("nan"))
            first_segment = False
            env.x, env.y, env.z = pg
            env.u, env.v, env.w = dg
            env.E = Eg
            env.alive = True
            env.steps = steps
            alive = True
            seg_fate = "cutoff"
            single_x.append(env.x); single_y.append(env.y); single_z.append(env.z)

            while alive and steps < max_steps and env.E > ecut:
                steps += 1
                mu = data.mu_total(env.E)
                if mu < ATTENUATION_FLOOR_CM_INV:
                    led["gamma_escaped"] += env.E
                    seg_fate = "escaped"; env.E = 0.0
                    alive = False
                    break
                dist = -math.log(random.random()) / mu
                env.x += dist * env.u
                env.y += dist * env.v
                env.z += dist * env.w

                if (env.x < env.xmin or env.x > env.xmax or
                    env.y < env.ymin or env.y > env.ymax or
                    env.z < env.zmin or env.z > env.zmax):
                    led["gamma_escaped"] += env.E
                    seg_fate = "escaped"; env.E = 0.0
                    alive = False
                    break
                photon_energy_in = env.E
                photon_incident_dir = np.array([env.u, env.v, env.w], dtype=float)
                new_dir, Eout, _secs, itype, _ = sample_photon_interaction(env.E, photon_incident_dir, data)
                angle = math.degrees(math.acos(np.clip(np.dot(photon_incident_dir, new_dir), -1, 1)))
                if (itype.startswith("photo") or itype.startswith("pair")) and _secs and _secs[0][0] == "electron":
                    e_dot = np.clip(np.dot(photon_incident_dir, _secs[0][2]), -1.0, 1.0)
                    angle = math.degrees(math.acos(e_dot))
                if itype in ("rayleigh", "compton"):
                    _th_rad = math.acos(np.clip(np.dot(photon_incident_dir, new_dir), -1, 1))
                    phi_deg_rec = math.degrees(
                        env._extract_local_phi(photon_incident_dir, new_dir, _th_rad))
                else:
                    phi_deg_rec = float("nan")
                if _secs and _secs[0][0] == "electron":
                    _e_dir = _secs[0][2]
                    _th_e  = math.acos(np.clip(np.dot(photon_incident_dir, _e_dir), -1.0, 1.0))
                    electron_phi_rec = math.degrees(
                        env._extract_local_phi(photon_incident_dir, _e_dir, _th_e))
                else:
                    electron_phi_rec = float("nan")
                interaction_record = {
                    "interaction": itype,
                    "free_path": dist,
                    "angle": angle,
                    "phi": phi_deg_rec,
                    "electron_phi": electron_phi_rec,
                    "position": (env.x, env.y, env.z),
                    "photon_energy_in": photon_energy_in,
                    "photon_energy_out": Eout,
                    "photon_incident_direction": photon_incident_dir,
                    "generation": gen,
                    "secondaries": _secs.copy()
                }
                photon_interactions.append(interaction_record)
                iid = len(photon_interactions) - 1
                # per-interaction energy bookkeeping (binding / pair rest mass)
                sec_E = sum(sc[1] for sc in _secs)
                if itype == "pair":
                    led["pair_restmass"] += 2.0 * mec2
                elif itype == "photo":
                    led["binding_dropped"] += max(photon_energy_in - Eout - sec_E, 0.0)
                # Bank every photon radiated by a lepton, including
                # bremsstrahlung and positron-annihilation photons.
                for sec in _secs:
                    if sec[0] in ("electron", "positron"):
                        _dep, _lept_secs, _lled = transport_lepton_csda(
                            sec[1], sec[2], (env.x, env.y, env.z), env.dose_tally, env,
                            Egrid=stopping_Egrid, S_col_vals=stopping_S_col,
                            S_rad_vals=stopping_S_rad,
                            is_positron=(sec[0] == "positron"), ecut=ecut)
                        led["dep_col"] += _lled["col_deposit"]
                        led["dep_cutoff"] += _lled["cutoff_deposit"]
                        led["dep_soft_rad"] += _lled["soft_rad_deposit"]
                        led["charged_escaped"] += _lled["escaped_kinetic"]
                        led["annih_created"] += _lled["annih_energy"]
                        for g in _lept_secs:
                            if g[0] != "photon":
                                continue
                            brec = {"tag": g[3], "E": float(g[1]),
                                    "dir": tuple(float(t) for t in g[2]),
                                    "birth": tuple(g[4]),
                                    "parent": sec[0], "parent_interaction": iid,
                                    "generation": gen + 1, "fate": "pending"}
                            banked_records.append(brec)
                            secondaries.append(("photon", g[1], g[2], g[3], g[4]))
                            if g[1] > ecut:
                                photon_bank.append((g[1], tuple(g[2]), g[4],
                                                    g[3], gen + 1, brec))
                            else:
                                led["subcutoff_dropped"] += g[1]
                                brec["fate"] = "subcutoff"
                secondaries.extend(_secs)
                single_x.append(env.x)
                single_y.append(env.y)
                single_z.append(env.z)
                env.u, env.v, env.w = new_dir
                env.E = Eout
                if env.E < ecut:
                    led["subcutoff_dropped"] += env.E
                    seg_fate = "absorbed"; env.E = 0.0
                    alive = False
            if alive and steps >= max_steps and env.E > ecut:
                led["truncated"] += env.E
                seg_fate = "truncated"
            if rec is not None:
                rec["fate"] = seg_fate

        # anything still banked when max_steps hit
        for Eg, dg, pg, tag, gen, rec in photon_bank:
            led["truncated"] += Eg
            if rec is not None:
                rec["fate"] = "truncated"
        sinks = (led["dep_col"] + led["dep_cutoff"] + led["dep_soft_rad"]
                 + led["gamma_escaped"] + led["charged_escaped"]
                 + led["subcutoff_dropped"] + led["binding_dropped"]
                 + led["pair_restmass"] + led["truncated"])
        led["closure_err"] = abs(led["E_in"] + led["annih_created"] - sinks)
        if led["closure_err"] > HISTORY_CLOSURE_ATOL_MEV:
            raise AssertionError(
                f"MC history {hist} energy closure failed: "
                f"{led['closure_err']:.3e} MeV")
        tracks.append({
            "coords": (single_x, single_y, single_z),
            "interactions": photon_interactions,
            "secondaries": secondaries,
            "banked_photons": banked_records,
            "ledger": led
        })
        all_secondaries.extend(secondaries)
        mc_interactions.extend(photon_interactions)
        if verbose and (hist + 1) % max(n_photons // 10, 1) == 0:
            print(f"   MC {hist + 1:>7d}/{n_photons} histories "
                  f"|{_ascii_bar(hist + 1, n_photons, 30)}| "
                  f"{len(mc_interactions)} interactions, "
                  f"worst closure {max(t['ledger']['closure_err'] for t in tracks):.1e}")
    return tracks, all_secondaries, env.dose_tally, mc_interactions


def _ascii_bar(v, vmax, width=40, ch="█"):
    n = 0 if vmax <= 0 else int(round(width * v / vmax))
    return ch * n


def run_learned_transport(policy, data, env, n_photons, E0=1.0,
                            max_steps=TRANSPORT_MAX_STEPS, ecut=DEFAULT_TRANSPORT_ECUT_MEV,
                            exec_logger=None, audit=None, device="cpu",
                            verbose=False, generator=None):
    """Transport photons using generated collision events and shared CSDA.

    Free paths remain analytical. Secondary photons, geometry, and energy
    ledgers follow the MC runner. Deposited energy is taken from the CSDA
    ledger, including collisional, cutoff, and soft-radiation deposits.
    Above-threshold transport requires trained pair-production heads.
    """
    pair_capable = bool(getattr(policy, "pair_heads_ready", False))
    if E0 >= PAIR_THRESHOLD_MEV and not pair_capable:
        raise RuntimeError(
            f"E0={E0} MeV is above the pair threshold but the loaded kernel "
            f"has no trained pair heads. Load a trained current checkpoint "
            f"before above-threshold evaluation.")
    stopping_Egrid, stopping_S_col, stopping_S_rad = \
        load_stopping_power(ELECTRON_STOPPING_POWER_TABLE)
    tracks, all_secs, interactions = [], [], []
    dose = np.zeros_like(np.asarray(env.dose_tally, dtype=float))
    audit = exec_logger.audit if exec_logger is not None else audit
    eid = 0
    bind = dict(PHOTO_SHELL_BINDINGS_MEV)
    for pid in range(n_photons):
        x0 = (random.random() * SOURCE_FIELD_WIDTH_CM) - SOURCE_FIELD_HALF_WIDTH_CM          # identical source model
        y0 = (random.random() * SOURCE_FIELD_WIDTH_CM) - SOURCE_FIELD_HALF_WIDTH_CM
        # bank entry: (E, dir, pos, tag, generation, record-or-None)
        photon_bank = [(E0, (0.0, 0.0, 1.0), (x0, y0, 0.0), "primary", 0, None)]
        step = 0
        sx, sy, sz = [], [], []
        photon_interactions, secondaries, banked_records = [], [], []
        E_dep = 0.0; fate = "cutoff"; E_esc = 0.0
        primary_terminated = False
        first_segment = True
        led = {"E_in": float(E0), "dep_col": 0.0, "dep_cutoff": 0.0,
               "dep_soft_rad": 0.0, "gamma_escaped": 0.0, "charged_escaped": 0.0,
               "subcutoff_dropped": 0.0, "binding_dropped": 0.0,
               "pair_restmass": 0.0, "annih_created": 0.0, "truncated": 0.0}

        while photon_bank and step < max_steps:
            E, (u, v, w), (x, y, z), tag, gen, rec = photon_bank.pop()
            if E <= ecut:
                led["subcutoff_dropped"] += E
                if rec is not None:
                    rec["fate"] = "subcutoff"
                continue
            if not (env.xmin <= x <= env.xmax and env.ymin <= y <= env.ymax
                    and env.zmin <= z <= env.zmax):
                led["gamma_escaped"] += E
                if rec is not None:
                    rec["fate"] = "born_outside"
                continue
            if not first_segment:                    # NaN track separator
                sx.append(float("nan")); sy.append(float("nan")); sz.append(float("nan"))
            first_segment = False
            sx.append(x); sy.append(y); sz.append(z)
            seg_fate = "cutoff"
            alive = True

            while alive and step < max_steps and E > ecut:
                mu = data.mu_total(E)
                if mu < ATTENUATION_FLOOR_CM_INV:
                    led["gamma_escaped"] += E
                    if not primary_terminated and gen == 0:
                        fate = "escaped"; E_esc = E; primary_terminated = True
                    seg_fate = "escaped"; E = 0.0; alive = False
                    break
                fp = -math.log(max(random.random(), FREE_PATH_RANDOM_FLOOR)) / mu
                nx, ny, nz = x + fp * u, y + fp * v, z + fp * w
                if not (env.xmin <= nx <= env.xmax and env.ymin <= ny <= env.ymax
                        and env.zmin <= nz <= env.zmax):
                    led["gamma_escaped"] += E
                    if not primary_terminated and gen == 0:
                        fate = "escaped"; E_esc = E; primary_terminated = True
                    seg_fate = "escaped"; E = 0.0; alive = False
                    break
                x, y, z = nx, ny, nz
                ev = policy.generate_event(
                    encode_policy_energy(E, env), generator=generator, audit=audit)
                th_r = math.radians(ev.theta_deg); ph_r = math.radians(ev.phi_deg)
                inc_dir = np.array([u, v, w], dtype=float)
                secs = []; comp_err = 0.0
                pname = PROCESS_NAMES[ev.process]
                if pname == "compton":
                    E_out = E / (1.0 + (E / mec2) * (1.0 - math.cos(th_r)))
                    new_dir = rotate_direction(inc_dir, th_r, ph_r)
                    pe = inc_dir * E - new_dir * E_out
                    nrm = np.linalg.norm(pe)
                    e_dir = pe / nrm if nrm > LEARNED_RECOIL_MOMENTUM_FLOOR_MEV_C else -new_dir
                    E_e = E - E_out
                    secs = [("electron", E_e, e_dir, "compton_e")]
                    rec_angle = ev.theta_deg
                    phi_rec = ev.phi_deg
                elif pname == "rayleigh":
                    E_out = E
                    new_dir = rotate_direction(inc_dir, th_r, ph_r)
                    rec_angle = ev.theta_deg
                    phi_rec = ev.phi_deg
                elif pname == "photo":
                    shell_name = SHELL_NAMES[ev.shell]
                    E_e = max(E - bind[shell_name], 0.0)
                    e_dir = rotate_direction(inc_dir, th_r, ph_r)
                    secs = [("electron", E_e, e_dir, f"photo_{shell_name}")]
                    E_out = 0.0
                    new_dir = np.array([0.0, 0.0, 0.0])
                    rec_angle = ev.theta_deg          # ELECTRON angle (MC convention)
                    phi_rec = float("nan")
                    led["binding_dropped"] += bind[shell_name] if E > bind[shell_name] else E
                else:                                  # pair
                    if E < PAIR_THRESHOLD_MEV or ev.pair is None:
                        # The support mask excludes pair production below
                        # threshold; a violation indicates an invalid event.
                        if audit is not None:
                            audit.pair_below_threshold += 1
                        raise AssertionError(
                            f"pair sampled below threshold/without kinematics "
                            f"at E={E:.6f} MeV — support mask violated")
                    else:
                        # Pair kinematics use the learned fraction and angles.
                        p_frac, e_th, e_ph, p_th, p_ph = ev.pair
                        _avail = max(E - 2.0 * mec2, 0.0)
                        T_e = p_frac * _avail          # conservation by
                        T_p = _avail - T_e             # construction
                        d_e = rotate_direction(inc_dir, math.radians(e_th),
                                               math.radians(e_ph))
                        d_p = rotate_direction(inc_dir, math.radians(p_th),
                                               math.radians(p_ph))
                        secs = [("electron", T_e, d_e, "pair_e"),
                                ("positron", T_p, d_p, "pair_p")]
                        E_out = 0.0
                        new_dir = np.array([0.0, 0.0, 0.0])
                        rec_angle = e_th               # ELECTRON angle (MC convention)
                        phi_rec = float("nan")
                        led["pair_restmass"] += 2.0 * mec2
                if secs and secs[0][0] == "electron":
                    _th_e = math.acos(np.clip(np.dot(inc_dir, secs[0][2]), -1, 1))
                    electron_phi_rec = math.degrees(
                        env._extract_local_phi(inc_dir, secs[0][2], _th_e))
                else:
                    electron_phi_rec = float("nan")
                sec_E = sum(sc[1] for sc in secs)
                if pname == "photo":
                    ebal = abs(E - E_out - sec_E - bind[SHELL_NAMES[ev.shell]])
                elif pname == "pair" and secs:
                    ebal = abs(E - sec_E - 2.0 * mec2)
                else:
                    ebal = abs(E - E_out - sec_E)
                rrec = {"interaction": pname, "free_path": fp, "angle": rec_angle,
                        "phi": phi_rec, "electron_phi": electron_phi_rec,
                        "position": (x, y, z), "photon_energy_in": E,
                        "photon_energy_out": E_out,
                        "photon_incident_direction": inc_dir,
                        "generation": gen,
                        "secondaries": secs.copy()}
                photon_interactions.append(rrec); interactions.append(rrec)
                iid = len(photon_interactions) - 1
                for sc in secs:
                    if sc[0] in ("electron", "positron"):
                        _dep, _lept_secs, _lled = transport_lepton_csda(
                            sc[1], sc[2], (x, y, z), dose, env,
                            Egrid=stopping_Egrid, S_col_vals=stopping_S_col,
                            S_rad_vals=stopping_S_rad,
                            is_positron=(sc[0] == "positron"), ecut=ecut)
                        # The ledger excludes banked radiation from deposits.
                        E_dep += _dep
                        led["dep_col"] += _lled["col_deposit"]
                        led["dep_cutoff"] += _lled["cutoff_deposit"]
                        led["dep_soft_rad"] += _lled["soft_rad_deposit"]
                        led["charged_escaped"] += _lled["escaped_kinetic"]
                        led["annih_created"] += _lled["annih_energy"]
                        for g in _lept_secs:
                            if g[0] != "photon":
                                continue
                            brec = {"tag": g[3], "E": float(g[1]),
                                    "dir": tuple(float(t) for t in g[2]),
                                    "birth": tuple(g[4]),
                                    "parent": sc[0], "parent_interaction": iid,
                                    "generation": gen + 1, "fate": "pending"}
                            banked_records.append(brec)
                            secondaries.append(("photon", g[1], g[2], g[3], g[4]))
                            all_secs.append(("photon", g[1], g[2], g[3], g[4]))
                            if g[1] > ecut:
                                photon_bank.append((g[1], tuple(g[2]), g[4],
                                                    g[3], gen + 1, brec))
                            else:
                                led["subcutoff_dropped"] += g[1]
                                brec["fate"] = "subcutoff"
                secondaries.extend(secs); all_secs.extend(secs)
                sx.append(x); sy.append(y); sz.append(z)
                if exec_logger is not None:
                    exec_logger.log_event(event_id=eid, photon_id=pid,
                        k=len(photon_interactions) - 1, process=ev.process,
                        shell=ev.shell, E_in=E, free_path=fp,
                        theta_deg=ev.theta_deg, phi_deg=ev.phi_deg, E_out=E_out,
                        sec_E=sec_E, pos=(x, y, z), compton_err=comp_err,
                        ebal_err=ebal)
                eid += 1; step += 1
                u, v, w = float(new_dir[0]), float(new_dir[1]), float(new_dir[2])
                E = E_out
                if pname in ("photo", "pair") or E <= ecut:
                    led["subcutoff_dropped"] += E if E > 0 else 0.0
                    if not primary_terminated and gen == 0:
                        fate = "absorbed"; primary_terminated = True
                    seg_fate = "absorbed"; E = 0.0; alive = False
            if alive and step >= max_steps and E > ecut:
                led["truncated"] += E
                seg_fate = "truncated"
            if rec is not None:
                rec["fate"] = seg_fate
        for E_, dg_, pg_, tag_, gen_, rec_ in photon_bank:
            led["truncated"] += E_
            if rec_ is not None:
                rec_["fate"] = "truncated"
        if step >= max_steps and fate == "cutoff":
            fate = "max_steps"
        sinks = (led["dep_col"] + led["dep_cutoff"] + led["dep_soft_rad"]
                 + led["gamma_escaped"] + led["charged_escaped"]
                 + led["subcutoff_dropped"] + led["binding_dropped"]
                 + led["pair_restmass"] + led["truncated"])
        led["closure_err"] = abs(led["E_in"] + led["annih_created"] - sinks)
        if led["closure_err"] > HISTORY_CLOSURE_ATOL_MEV:
            raise AssertionError(
                f"agent history {pid} energy closure failed: "
                f"{led['closure_err']:.3e} MeV")
        tracks.append({"coords": (sx, sy, sz),
                       "interactions": photon_interactions,
                       "secondaries": secondaries,
                       "banked_photons": banked_records,
                       "ledger": led,
                       "n_interactions": len(photon_interactions)})
        if exec_logger is not None:
            exec_logger.log_photon(pid, len(photon_interactions), fate,
                                   E, E_dep, E_esc)
        if verbose and (pid + 1) % max(n_photons // 10, 1) == 0:
            print(f"   Agent {pid + 1:>7d}/{n_photons} histories "
                  f"|{_ascii_bar(pid + 1, n_photons, 30)}| "
                  f"{len(interactions)} interactions")
    return tracks, all_secs, dose, interactions


class _HistorySlot:
    """One in-flight primary history for the batched runner.  The scalar
    physics below — bank handling, boundary/cutoff logic, event decode,
    CSDA application, banking, records, ledger — is a line-faithful
    transcription of run_learned_transport's per-history body; only the
    neural sampling is lifted out into the pooled batch."""

    __slots__ = ("pid", "ctx", "bank", "step", "sx", "sy", "sz",
                 "photon_interactions", "secondaries", "banked_records",
                 "E_dep", "fate", "E_esc", "primary_terminated",
                 "first_segment", "led", "E", "u", "v", "w", "x", "y", "z",
                 "tag", "gen", "rec", "seg_fate", "inflight", "fp")

    def __init__(self, pid, E0, ctx):
        self.pid, self.ctx = pid, ctx
        x0 = (random.random() * SOURCE_FIELD_WIDTH_CM) - SOURCE_FIELD_HALF_WIDTH_CM          # identical source model
        y0 = (random.random() * SOURCE_FIELD_WIDTH_CM) - SOURCE_FIELD_HALF_WIDTH_CM
        self.bank = [(E0, (0.0, 0.0, 1.0), (x0, y0, 0.0), "primary", 0, None)]
        self.step = 0
        self.sx, self.sy, self.sz = [], [], []
        self.photon_interactions, self.secondaries = [], []
        self.banked_records = []
        self.E_dep = 0.0; self.fate = "cutoff"; self.E_esc = 0.0
        self.primary_terminated = False
        self.first_segment = True
        self.led = {"E_in": float(E0), "dep_col": 0.0, "dep_cutoff": 0.0,
                    "dep_soft_rad": 0.0, "gamma_escaped": 0.0,
                    "charged_escaped": 0.0, "subcutoff_dropped": 0.0,
                    "binding_dropped": 0.0, "pair_restmass": 0.0,
                    "annih_created": 0.0, "truncated": 0.0}
        self.inflight = False

    # ---- segment close helpers (mirror the serial inner-loop exits) ----
    def _close_segment(self, seg_fate):
        if self.rec is not None:
            self.rec["fate"] = seg_fate
        self.inflight = False

    def advance(self):
        """Run analytic transport (pop/cutoff/outside/free-flight/boundary)
        until this history is interaction-ready or finished.  Returns
        incident energy when inference is required, or None when the
        history is complete (caller then calls finalize())."""
        env, data, ecut, max_steps = (self.ctx["env"], self.ctx["data"],
                                      self.ctx["ecut"], self.ctx["max_steps"])
        while True:
            if not self.inflight:
                if not self.bank or self.step >= max_steps:
                    return None
                (self.E, (self.u, self.v, self.w),
                 (self.x, self.y, self.z), self.tag, self.gen,
                 self.rec) = self.bank.pop()
                if self.E <= ecut:
                    self.led["subcutoff_dropped"] += self.E
                    if self.rec is not None:
                        self.rec["fate"] = "subcutoff"
                    continue
                if not (env.xmin <= self.x <= env.xmax
                        and env.ymin <= self.y <= env.ymax
                        and env.zmin <= self.z <= env.zmax):
                    self.led["gamma_escaped"] += self.E
                    if self.rec is not None:
                        self.rec["fate"] = "born_outside"
                    continue
                if not self.first_segment:           # NaN track separator
                    self.sx.append(float("nan")); self.sy.append(float("nan"))
                    self.sz.append(float("nan"))
                self.first_segment = False
                self.sx.append(self.x); self.sy.append(self.y)
                self.sz.append(self.z)
                self.seg_fate = "cutoff"
                self.inflight = True
            # in-flight step attempt (serial inner-while head)
            if not (self.step < max_steps and self.E > ecut):
                if self.step >= max_steps and self.E > ecut:
                    self.led["truncated"] += self.E
                    self.seg_fate = "truncated"
                self._close_segment(self.seg_fate)
                continue
            mu = data.mu_total(self.E)
            if mu < ATTENUATION_FLOOR_CM_INV:
                self.led["gamma_escaped"] += self.E
                if not self.primary_terminated and self.gen == 0:
                    self.fate = "escaped"; self.E_esc = self.E
                    self.primary_terminated = True
                self.seg_fate = "escaped"; self.E = 0.0
                self._close_segment(self.seg_fate)
                continue
            fp = -math.log(max(random.random(), FREE_PATH_RANDOM_FLOOR)) / mu
            nx = self.x + fp * self.u
            ny = self.y + fp * self.v
            nz = self.z + fp * self.w
            if not (env.xmin <= nx <= env.xmax and env.ymin <= ny <= env.ymax
                    and env.zmin <= nz <= env.zmax):
                self.led["gamma_escaped"] += self.E
                if not self.primary_terminated and self.gen == 0:
                    self.fate = "escaped"; self.E_esc = self.E
                    self.primary_terminated = True
                self.seg_fate = "escaped"; self.E = 0.0
                self._close_segment(self.seg_fate)
                continue
            self.x, self.y, self.z = nx, ny, nz
            self.fp = fp
            return self.E

    def apply(self, process, shell, theta_deg, phi_deg, pair_tuple):
        """Decode and apply ONE sampled event — verbatim serial physics."""
        env, data = self.ctx["env"], self.ctx["data"]
        ecut = self.ctx["ecut"]
        dose = self.ctx["dose"]; bind = self.ctx["bind"]
        audit = self.ctx["audit"]
        interactions, all_secs = (self.ctx["interactions"],
                                  self.ctx["all_secs"])
        E, fp = self.E, self.fp
        x, y, z = self.x, self.y, self.z
        u, v, w = self.u, self.v, self.w
        gen = self.gen
        th_r = math.radians(theta_deg); ph_r = math.radians(phi_deg)
        inc_dir = np.array([u, v, w], dtype=float)
        secs = []; comp_err = 0.0
        pname = PROCESS_NAMES[process]
        if pname == "compton":
            E_out = E / (1.0 + (E / mec2) * (1.0 - math.cos(th_r)))
            new_dir = rotate_direction(inc_dir, th_r, ph_r)
            pe = inc_dir * E - new_dir * E_out
            nrm = np.linalg.norm(pe)
            e_dir = pe / nrm if nrm > LEARNED_RECOIL_MOMENTUM_FLOOR_MEV_C else -new_dir
            E_e = E - E_out
            secs = [("electron", E_e, e_dir, "compton_e")]
            rec_angle = theta_deg
            phi_rec = phi_deg
        elif pname == "rayleigh":
            E_out = E
            new_dir = rotate_direction(inc_dir, th_r, ph_r)
            rec_angle = theta_deg
            phi_rec = phi_deg
        elif pname == "photo":
            shell_name = SHELL_NAMES[shell]
            E_e = max(E - bind[shell_name], 0.0)
            e_dir = rotate_direction(inc_dir, th_r, ph_r)
            secs = [("electron", E_e, e_dir, f"photo_{shell_name}")]
            E_out = 0.0
            new_dir = np.array([0.0, 0.0, 0.0])
            rec_angle = theta_deg          # ELECTRON angle (MC convention)
            phi_rec = float("nan")
            self.led["binding_dropped"] += (bind[shell_name]
                                            if E > bind[shell_name] else E)
        else:                              # pair
            if E < PAIR_THRESHOLD_MEV or pair_tuple is None:
                if audit is not None:
                    audit.pair_below_threshold += 1
                raise AssertionError(
                    f"pair sampled below threshold/without kinematics "
                    f"at E={E:.6f} MeV — support mask violated")
            p_frac, e_th, e_ph, p_th, p_ph = pair_tuple
            _avail = max(E - 2.0 * mec2, 0.0)
            T_e = p_frac * _avail
            T_p = _avail - T_e
            d_e = rotate_direction(inc_dir, math.radians(e_th),
                                   math.radians(e_ph))
            d_p = rotate_direction(inc_dir, math.radians(p_th),
                                   math.radians(p_ph))
            secs = [("electron", T_e, d_e, "pair_e"),
                    ("positron", T_p, d_p, "pair_p")]
            E_out = 0.0
            new_dir = np.array([0.0, 0.0, 0.0])
            rec_angle = e_th               # ELECTRON angle (MC convention)
            phi_rec = float("nan")
            self.led["pair_restmass"] += 2.0 * mec2
        if secs and secs[0][0] == "electron":
            _th_e = math.acos(np.clip(np.dot(inc_dir, secs[0][2]), -1, 1))
            electron_phi_rec = math.degrees(
                env._extract_local_phi(inc_dir, secs[0][2], _th_e))
        else:
            electron_phi_rec = float("nan")
        sec_E = sum(sc[1] for sc in secs)
        if pname == "photo":
            ebal = abs(E - E_out - sec_E - bind[SHELL_NAMES[shell]])
        elif pname == "pair" and secs:
            ebal = abs(E - sec_E - 2.0 * mec2)
        else:
            ebal = abs(E - E_out - sec_E)
        rrec = {"interaction": pname, "free_path": fp, "angle": rec_angle,
                "phi": phi_rec, "electron_phi": electron_phi_rec,
                "position": (x, y, z), "photon_energy_in": E,
                "photon_energy_out": E_out,
                "photon_incident_direction": inc_dir,
                "generation": gen,
                "secondaries": secs.copy()}
        self.photon_interactions.append(rrec); interactions.append(rrec)
        iid = len(self.photon_interactions) - 1
        for sc in secs:
            if sc[0] in ("electron", "positron"):
                _dep, _lept_secs, _lled = transport_lepton_csda(
                    sc[1], sc[2], (x, y, z), dose, env,
                    Egrid=self.ctx["stop_E"],
                    S_col_vals=self.ctx["stop_col"],
                    S_rad_vals=self.ctx["stop_rad"],
                    is_positron=(sc[0] == "positron"), ecut=ecut)
                self.E_dep += _dep
                self.led["dep_col"] += _lled["col_deposit"]
                self.led["dep_cutoff"] += _lled["cutoff_deposit"]
                self.led["dep_soft_rad"] += _lled["soft_rad_deposit"]
                self.led["charged_escaped"] += _lled["escaped_kinetic"]
                self.led["annih_created"] += _lled["annih_energy"]
                for g in _lept_secs:
                    if g[0] != "photon":
                        continue
                    brec = {"tag": g[3], "E": float(g[1]),
                            "dir": tuple(float(t) for t in g[2]),
                            "birth": tuple(g[4]),
                            "parent": sc[0], "parent_interaction": iid,
                            "generation": gen + 1, "fate": "pending"}
                    self.banked_records.append(brec)
                    self.secondaries.append(("photon", g[1], g[2], g[3],
                                             g[4]))
                    all_secs.append(("photon", g[1], g[2], g[3], g[4]))
                    if g[1] > ecut:
                        self.bank.append((g[1], tuple(g[2]), g[4],
                                          g[3], gen + 1, brec))
                    else:
                        self.led["subcutoff_dropped"] += g[1]
                        brec["fate"] = "subcutoff"
        self.secondaries.extend(secs); all_secs.extend(secs)
        self.sx.append(x); self.sy.append(y); self.sz.append(z)
        self.step += 1
        self.u = float(new_dir[0]); self.v = float(new_dir[1])
        self.w = float(new_dir[2])
        self.E = E_out
        if pname in ("photo", "pair") or self.E <= ecut:
            self.led["subcutoff_dropped"] += self.E if self.E > 0 else 0.0
            if not self.primary_terminated and gen == 0:
                self.fate = "absorbed"; self.primary_terminated = True
            self.seg_fate = "absorbed"; self.E = 0.0
            self._close_segment(self.seg_fate)

    def finalize(self):
        """Serial post-history block: leftover bank, fate, closure assert."""
        for E_, dg_, pg_, tag_, gen_, rec_ in self.bank:
            self.led["truncated"] += E_
            if rec_ is not None:
                rec_["fate"] = "truncated"
        if self.step >= self.ctx["max_steps"] and self.fate == "cutoff":
            self.fate = "max_steps"
        led = self.led
        sinks = (led["dep_col"] + led["dep_cutoff"] + led["dep_soft_rad"]
                 + led["gamma_escaped"] + led["charged_escaped"]
                 + led["subcutoff_dropped"] + led["binding_dropped"]
                 + led["pair_restmass"] + led["truncated"])
        led["closure_err"] = abs(led["E_in"] + led["annih_created"] - sinks)
        if led["closure_err"] > HISTORY_CLOSURE_ATOL_MEV:
            raise AssertionError(
                f"agent history {self.pid} energy closure failed: "
                f"{led['closure_err']:.3e} MeV")
        return {"coords": (self.sx, self.sy, self.sz),
                "interactions": self.photon_interactions,
                "secondaries": self.secondaries,
                "banked_photons": self.banked_records,
                "ledger": led,
                "n_interactions": len(self.photon_interactions)}


def run_learned_transport_batched(policy, data, env, n_photons, E0=1.0,
                                    max_steps=TRANSPORT_MAX_STEPS, ecut=DEFAULT_TRANSPORT_ECUT_MEV,
                                    audit=None,
                                    device="cpu", inference_batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
                                    generator=None, verbose=False):
    """Pool independent photon histories for batched event generation.

    Each wave advances histories to their next collision, generates events
    from the incident energies, and applies the same scalar physics as the
    serial runner. Batching changes the order of random draws, so seeded
    serial and batched runs need not produce identical histories.

    Returns ``(tracks, all_secs, dose, interactions, perf)``. In ``perf``,
    ``t_energy_encoding`` measures energy encoding; ``t_event_generation``
    includes the device transfer performed inside the policy.
    """
    pair_capable = bool(getattr(policy, "pair_heads_ready", False))
    if E0 >= PAIR_THRESHOLD_MEV and not pair_capable:
        raise RuntimeError(
            f"E0={E0} MeV is above the pair threshold but the loaded kernel "
            f"has no trained pair heads.")
    stop_E, stop_col, stop_rad = \
        load_stopping_power(ELECTRON_STOPPING_POWER_TABLE)
    tracks, all_secs, interactions = [], [], []
    dose = np.zeros_like(np.asarray(env.dose_tally, dtype=float))
    ctx = {"env": env, "data": data, "ecut": ecut, "max_steps": max_steps,
           "dose": dose, "interactions": interactions, "all_secs": all_secs,
           "audit": audit, "stop_E": stop_E, "stop_col": stop_col,
           "stop_rad": stop_rad,
           "bind": dict(PHOTO_SHELL_BINDINGS_MEV)}
    _is_cuda = str(device).startswith("cuda")
    if _is_cuda:
        import torch
    perf = {"t_advance": 0.0, "t_energy_encoding": 0.0, "t_event_generation": 0.0,
            "t_apply": 0.0, "waves": 0, "rows": 0,
            "fill_min": None, "fill_sum": 0}
    slots, next_pid, completed = [], 0, 0
    t_total0 = time.time()
    while completed < n_photons:
        while len(slots) < inference_batch_size and next_pid < n_photons:
            slots.append(_HistorySlot(next_pid, E0, ctx))
            next_pid += 1
        t0 = time.time()
        ready, states, survivors = [], [], []
        for sl in slots:
            st = sl.advance()
            if st is None:
                tracks.append(sl.finalize())
                completed += 1
            else:
                survivors.append(sl); ready.append(sl); states.append(st)
        slots = survivors
        perf["t_advance"] += time.time() - t0
        if not ready:
            if next_pid >= n_photons and not slots:
                break
            continue
        t0 = time.time()
        energies = encode_policy_energy_batch(states, env)
        perf["t_energy_encoding"] += time.time() - t0
        t0 = time.time()
        evd = policy.generate_events_batch(
            energies, audit=audit, generator=generator)
        if _is_cuda:
            torch.cuda.synchronize()
        perf["t_event_generation"] += time.time() - t0
        Bn = len(ready)
        perf["waves"] += 1; perf["rows"] += Bn
        perf["fill_sum"] += Bn
        perf["fill_min"] = (Bn if perf["fill_min"] is None
                            else min(perf["fill_min"], Bn))
        t0 = time.time()
        for i, sl in enumerate(ready):
            pt = (None if math.isnan(evd["pair_frac"][i]) else
                  (float(evd["pair_frac"][i]),
                   float(evd["pair_e_theta"][i]),
                   float(evd["pair_e_phi"][i]),
                   float(evd["pair_p_theta"][i]),
                   float(evd["pair_p_phi"][i])))
            sl.apply(int(evd["process"][i]), int(evd["shell"][i]),
                     float(evd["theta_deg"][i]), float(evd["phi_deg"][i]),
                     pt)
        perf["t_apply"] += time.time() - t0
    perf["t_total"] = time.time() - t_total0
    perf["fill_mean"] = perf["rows"] / max(perf["waves"], 1)
    perf["hist_s"] = n_photons / max(perf["t_total"], PERFORMANCE_TIME_FLOOR_S)
    if _is_cuda:
        perf["peak_gpu_mem_MB"] = float(
            torch.cuda.max_memory_allocated() / 1e6)
    return tracks, all_secs, dose, interactions, perf
