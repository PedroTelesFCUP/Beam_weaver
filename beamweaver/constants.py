"""Physical constants, unit conversions and current data/model configuration.

Rest energies are in MeV. Reference sampling, learned reconstruction,
coordinate transforms and pair support share the canonical ``mec2``.
Equal-valued settings with different meanings remain independent.
"""

# Copyright (c) 2025–2026 Pedro Teles. Apache-2.0.

import numpy as np

BW_VERSION = "0.4.1"

mec2 = 0.51099895069  # Electron rest energy, MeV.
PAIR_THRESHOLD_MEV = 2.0 * mec2

HC_KEV_A = 12.3984

KEV_PER_MEV = 1000.0
MEV_PER_EV = 1e-6
HC_MEV_A = HC_KEV_A / KEV_PER_MEV

PHOTO_SHELL_BINDINGS_EV = {
    "H_K": 13.6,
    "O_K": 532.0,
    "O_L1": 40.0,
    "O_L2": 17.0,
    "O_L3": 17.0,
}
PHOTO_SHELL_BINDINGS_MEV = {
    shell: energy * MEV_PER_EV
    for shell, energy in PHOTO_SHELL_BINDINGS_EV.items()
}

PROCESS_NAMES = ["rayleigh", "compton", "photo", "pair"]

SHELL_NAMES = ["H_K", "O_K", "O_L1", "O_L2", "O_L3"]
N_PROCESSES = len(PROCESS_NAMES)
N_SHELLS = len(SHELL_NAMES)
PROCESS_INDEX = {name: i for i, name in enumerate(PROCESS_NAMES)}
SHELL_INDEX = {name: i for i, name in enumerate(SHELL_NAMES)}


# Transport defaults are independent of the learned energy domain.
DEFAULT_TRANSPORT_ECUT_MEV = 0.001
TRANSPORT_MAX_STEPS = 100_000
SOURCE_FIELD_WIDTH_CM = 10.0
SOURCE_FIELD_HALF_WIDTH_CM = SOURCE_FIELD_WIDTH_CM / 2.0

# Each numerical guard has its own meaning and units.
DIRECTION_FRAME_SWITCH = 0.9999
HISTORY_CLOSURE_ATOL_MEV = 2e-6
ATTENUATION_FLOOR_CM_INV = 1e-30
SHELL_CROSS_SECTION_FLOOR_CM2_G = 1e-30
FREE_PATH_RANDOM_FLOOR = 1e-300
LEARNED_RECOIL_MOMENTUM_FLOOR_MEV_C = 1e-12
EVENT_ENERGY_BALANCE_ATOL_MEV = 1e-3
NEGATIVE_ENERGY_ATOL_MEV = 1e-12
PERFORMANCE_TIME_FLOOR_S = 1e-9

# File names are shared; each workflow selects the tables it requires.
PHOTON_CROSS_SECTION_TABLE = "Final_cross_sections.csv"
RAYLEIGH_CROSS_SECTION_TABLE = "Rayleigh_cross_sections.csv"
PHOTO_SHELL_TABLE = "WaterPhotoShells.csv"
COHERENT_FORM_FACTOR_TABLE = "water_fq.csv"
INCOHERENT_SCATTERING_TABLE = "water_sq.csv"
ELECTRON_STOPPING_POWER_TABLE = "ElectronStoppingPower.csv"

# Equal counts for different workloads remain independent controls.
DEFAULT_TRAIN_BATCH_SIZE = 8192
DEFAULT_INFERENCE_BATCH_SIZE = 256
DEFAULT_FACTOR_VALIDATION_SAMPLES = 8192
DEFAULT_REFERENCE_VALIDATION_SAMPLES = 100_000
REFERENCE_POLAR_GRID_POINTS = 180
DEFAULT_RUN_ENERGY_MEV = 1.0
DEFAULT_RUN_HISTORIES = 500
DEFAULT_COMPARISON_ENERGIES = (0.05, 0.5, 1.0, 5.0)
DEFAULT_COMPARISON_HISTORIES = 2000

_PAIR_ALPHA_FS = 1.0 / 137.035999177

_PAIR_R_TABLE = {1: 122.81, 2: 73.167, 3: 69.228, 4: 67.301, 5: 64.696,
                 6: 61.228, 7: 57.524, 8: 54.033, 9: 50.787, 10: 47.851}

_PAIR_Z_EQ = 7

_PAIR_R_EQ = _PAIR_R_TABLE[_PAIR_Z_EQ]

BW4_EMIN_MEV, BW4_EMAX_MEV = 0.001, 10.0

BW4_NORM_LOGE_SPEC = (
    f"x = 2*(log10(E/{BW4_EMIN_MEV:g})/"
    f"log10({BW4_EMAX_MEV / BW4_EMIN_MEV:g})) - 1 on "
    f"[{BW4_EMIN_MEV:g},{BW4_EMAX_MEV:g}] MeV -> [-1,1]")

BW4_PAIR_TRAIN_EXTRA = [1.023, 1.03, 1.05, 1.075, 1.10]

BW4_TEST_E = [0.010, 0.020, 0.050, 0.100, 0.5, 1.0, 2.0, 5.0, 10.0]

BW4_K = {"ray_q2": 720, "comp_u": 180, "photo_nu": 180, "phi": 36,
         "pair_frac": 90, "pair_nu": 180}

BW4_M_DEFAULT = {"process": 32768, "shell": 32768, "ray": 8192,
                 "comp": 8192, "photo": 8192, "pair": 16384}

BW4_SANITY_FLOOR = 0.02

BW4_PHI_EPS = 1e-6

BW4_S_SUB = 8

BW4_COORDS = {
    "ray_q2": "s=ln(q2/q2_max)=ln((1-cos t)/2), support (-inf,0] for ALL E "
              "(log reduced momentum transfer; Compton-u analogue); "
              "cos t = 1-2*exp(s); q2 = 4k^2*exp(s), k=E_keV/HC_KEV_A",
    "comp_u": "u=(1-tau)/(1-tau_min); tau=E'/E; tau_min=1/(1+2E/mec2); "
              "cos t = 1-(1-tau)/((E/mec2)tau)",
    "photo_nu": "nu=1-cos t in [0,2]",
    "pair": "f=T-/(T-+T+); nu±=1-cos t±; phi± independent; lepton input "
            "f_lep=T_lep/(E-2mec2)",
    "phi": (f"{BW4_K['phi']} equal circular bins on [0,2pi); valid iff sin t>"
            + format(BW4_PHI_EPS, "g").replace("e-0", "e-"))}

_BW4_Z = np.array([0.0, 0.0, 1.0])

BW4_FACTOR_HEADS = {
    "process": ("process_head", 1, N_PROCESSES),
    "shell": ("shell_head", 1, N_SHELLS),
    "ray_q2": ("rayleigh_q2_head", 1, None),
    "ray_phi": ("rayleigh_phi_head", 0, None),
    "comp_u": ("compton_u_head", 1, None),
    "comp_phi": ("compton_phi_head", 0, None),
    "photo_nu": ("photo_nu_head", 1 + N_SHELLS, None),
    "photo_phi": ("photo_phi_head", N_SHELLS, None),
    "pair_frac": ("pair_fraction_head", 1, None),
    "pair_e_polar": ("pair_electron_polar_head", 2, None),
    "pair_e_phi": ("pair_electron_phi_head", 2, None),
    "pair_p_polar": ("pair_positron_polar_head", 2, None),
    "pair_p_phi": ("pair_positron_phi_head", 2, None)}

BW4_TRAIN_SEQUENCE = [
    ("process",), ("shell",), ("ray_q2",), ("ray_phi",), ("comp_u",),
    ("comp_phi",), ("photo_nu",), ("photo_phi",), ("pair_frac",),
    ("pair_e_polar", "pair_e_phi", "pair_p_polar", "pair_p_phi")]

CURRENT_POLICY_SCHEMA = "beam_weaver_generative_policy_v040"


def runtime_physics_metadata():
    """Return fresh numerical provenance for the current reconstruction code.

    This describes this runtime, not the training history of loaded weights.
    """
    return {
        "mec2_MeV": mec2,
        "pair_threshold_MeV": PAIR_THRESHOLD_MEV,
        "hc_keV_angstrom": HC_KEV_A,
        "hc_MeV_angstrom": HC_MEV_A,
        "shell_bindings_eV": dict(PHOTO_SHELL_BINDINGS_EV),
    }
