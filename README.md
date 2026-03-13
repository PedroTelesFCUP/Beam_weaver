# Rad-SAC Agent

**Artificial Intelligence-Driven Radiation Transport: A Soft Actor–Critic Agent for Photon Transport in Water**

> Research software for learning photon transport in water with a custom hybrid Soft Actor–Critic (SAC) agent, a Monte Carlo reference simulator, and evaluation utilities for shower tracks and percentage depth dose (PDD).

**Status:** proof of concept / research code  
**DOI:** pending first Zenodo release  
**License:** Apache-2.0

## What this repository contains

This repository packages the current research code for a custom radiation-transport reinforcement learning workflow centered on a single Python script:

- `Rad_sac_24_maio_fixed.py` — main script containing:
  - Monte Carlo photon interaction sampling
  - a custom `WaterPhotonHybridEnvPenelope` environment
  - a phased reward wrapper and curriculum schedule
  - a custom n-step SAC implementation and policy
  - physics-head pretraining utilities
  - evaluation utilities for track visualization, secondary analysis, and PDD comparison

The current implementation models four photon interaction channels in water:

1. Rayleigh scattering
2. Compton scattering
3. Photoelectric absorption
4. Pair production

## Research scope

This codebase accompanies a proof-of-concept study showing that a SAC agent can learn physically informed photon transport behavior in a homogeneous water phantom, using a simplified but physics-grounded Monte Carlo simulator as the reference environment.

This is **research software**, not a clinical tool and not a certified medical device. It must **not** be used for patient-specific treatment planning or clinical decision-making without extensive independent validation, QA, and regulatory review.

## Main features

- Custom hybrid SAC policy for mixed discrete/continuous action prediction
- Physics-aware observation design
- Curriculum training across multiple phases
- Physics-head pretraining from Monte Carlo-generated samples
- Support for MC-vs-agent comparison of:
  - interaction histories
  - secondary particles
  - track geometry
  - depth-dose behavior
- Evaluation helpers for fixed-energy photon beams in a 100 × 100 × 100 cm water phantom

## Required input files

The current script expects several external data files in the **working directory** (repository root unless you change the code paths):

### Required at startup
- `NIST_WaterCrossSections.csv`
- `OxygenPhotoShells.csv`
- `ElectronStoppingPower.csv`

### Also used by training / evaluation / pretraining paths
- `Final_cross_sections.csv`
- `Rayleigh_cross_sections.csv`
- `mc_physics_data.npz` (if using physics-head pretraining from a saved dataset)
- `physics_head_pretrained.pth` (optional pretrained weights)
- `replay_buffer.pkl` (needed by the current eval path after loading a trained model)

The code also writes multiple runtime artifacts such as model checkpoints, TensorBoard logs, cached statistics, and profiling outputs.

## Quick start

### 1. Create a clean environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate rad-sac-agent
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add the data files

Place the CSV / NPZ / checkpoint files listed above in the repository root, or edit the file paths in `Rad_sac_24_maio_fixed.py` to point to your preferred layout.

A helper note is included in `data/README.md`.

### 3. Run the script

```bash
python Rad_sac_24_maio_fixed.py
```

The script is interactive and will ask whether you want to:

- generate an MC dataset
- pretrain the physics head
- train a new model or evaluate an existing one
- optionally profile the run

## Typical workflow

### Generate MC pretraining data
Start the script and answer **yes** to dataset generation. This creates:

- `mc_physics_data.npz`

### Pretrain the physics head
Start the script and answer **yes** to physics-head pretraining. This creates:

- `physics_head_pretrained.pth`

### Train a policy
Start the script and choose:

- `train`
- total timesteps
- evaluation energy in MeV

This will produce artifacts such as:

- `hybrid_sac_model.zip`
- `replay_buffer.pkl`
- `histograms.pkl`
- `acc_stats.pkl`
- `dist_stats.pkl`
- `tb_logs/`
- `older/`

### Evaluate a trained policy
Start the script and choose:

- `eval`
- fixed photon energy in MeV

The current evaluation path expects an existing `hybrid_sac_model.zip` and `replay_buffer.pkl`.

## Suggested repository layout

A practical layout for future cleanup is:

```text
rad-sac-agent/
├── Rad_sac_24_maio_fixed.py
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── requirements.txt
├── environment.yml
├── data/
│   └── README.md
├── models/
├── outputs/
└── docs/
```

At present, the script still uses root-level relative file paths. A later refactor should move file handling into a proper config system.

## Reproducibility notes

For a cleaner public release, consider next:

- rename `Rad_sac_24_maio_fixed.py` to a stable project name such as `rad_sac_agent.py`
- split the monolithic script into modules:
  - `physics.py`
  - `env.py`
  - `policy.py`
  - `train.py`
  - `eval.py`
  - `utils.py`
- add a fixed random-seed interface
- record exact package versions used in the published experiments
- export one small example dataset for smoke testing
- add one non-interactive CLI for automation

## Citation

Please cite the archived software release once the Zenodo DOI is minted.

A machine-readable citation stub is provided in `CITATION.cff`.  
Zenodo metadata for release archiving is provided in `.zenodo.json`.

## License

This repository’s original code is released under the **Apache License 2.0**.

Third-party data, external manuals, and dependencies remain under their own respective licenses and terms. See `LICENSE-THIRD-PARTY.md`.

## Acknowledgment / provenance

This implementation is inspired by standard radiation transport literature and by comparisons reported against PENELOPE, but this repository should be presented as an **independent Python research implementation**. Do **not** imply that it redistributes PENELOPE source code or official PENELOPE data unless you explicitly have permission and include them under their original terms.
