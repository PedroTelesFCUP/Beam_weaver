# Beam Weaver

**Physics-informed reinforcement learning for stochastic photon transport in water**

> Beam Weaver is a research software prototype for learning event-by-event photon transport in a homogeneous water phantom. It combines a custom Monte Carlo reference simulator, a physics-informed reinforcement learning environment, and evaluation tools that compare learned particle histories against conventional transport outputs such as interaction statistics, secondary production, particle tracks, and percentage depth dose (PDD).

**Status:** research prototype / paused snapshot  
**DOI:** pending first Zenodo release  
**License:** Apache-2.0  
**Primary domain:** radiation transport and AI research

**Authors:**  
- **Pedro Teles** — Assistant Professor, Department of Physics and Astronomy, Faculty of Sciences, University of Porto  
- **João Melo** — MSc Student in Medical Physics, Department of Physics and Astronomy, Faculty of Sciences, University of Porto

## Research-use notice

Beam Weaver is released **for research purposes only**.

No part of this repository should be used for clinical, industrial, regulatory, commercial, or other safety-critical purposes. This applies to **all** components of the project, including the reinforcement-learning models, evaluation scripts, generated outputs, and the **in-built Monte Carlo simulator**.

The code is provided as a research prototype for method development, inspection, reproducibility, and academic discussion. It must not be represented as validated for real-world decision-making outside research settings.

## What Beam Weaver does

Beam Weaver explores whether a **Soft Actor–Critic (SAC)** agent can learn the stochastic logic of photon transport in water at the **event level**, rather than only predicting final observables.

Instead of treating transport as a black-box regression problem, Beam Weaver models the transport process step by step. For each photon history, the code is designed to predict quantities such as:

- the next interaction type
- the distance to the next interaction
- scattering geometry
- the outgoing photon state
- the number, energy, and direction of secondary particles

This makes Beam Weaver closer to a **learned transport-engine prototype** than to a generic machine-learning regressor.

## Current project status

Beam Weaver's central purpose of the project was largely achieved: the code can already simulate the main photon interaction processes in a physically meaningful way and support comparison between the learned agent and the Monte Carlo reference workflow. The main remaining development task  is  **fine-tuning the angular predictions / angular sampling behavior**, which is the principal unfinished component at this time.

In other words, this project is a working research prototype with a specific remaining refinement target.

## Scientific scope

The current implementation studies photon transport in a **homogeneous water phantom** and packages a single main research script, `Beam_weaver.py`, which contains:

- a custom Monte Carlo-style photon transport simulator in water
- a custom `WaterPhotonHybridEnvPenelope` environment
- a hybrid SAC actor-critic implementation for mixed discrete/continuous prediction
- curriculum-based training logic with phase switching
- physics-head pretraining on Monte Carlo-generated targets
- evaluation utilities for comparing Monte Carlo and agent-generated showers

In its current form, Beam Weaver models four photon interaction channels in water:

1. Rayleigh scattering  
2. Compton scattering  
3. Photoelectric absorption  
4. Pair production

The evaluation workflow supports comparison between Monte Carlo reference behavior and agent behavior through:

- interaction frequencies
- secondary-particle generation
- track-level shower behavior
- angular behavior
- percentage depth dose (PDD) curves

## Why this software exists

Monte Carlo radiation transport remains the reference approach for detailed stochastic transport simulation, but it is often computationally expensive. Beam Weaver was created as a proof-of-concept framework for studying whether reinforcement learning can reproduce key parts of the photon transport loop while remaining physically grounded.

The project therefore sits at the intersection of:

- radiation transport simulation
- reinforcement learning
- physics-informed machine learning
- surrogate modeling for Monte Carlo workflows

## What Beam Weaver does **not** claim

Beam Weaver is **not** a validated Monte Carlo engine, or a completed production codebase.


Its purpose is research: to investigate whether a physics-informed RL agent can learn the structure of stochastic photon transport from a Monte Carlo reference process.

## Repository contents

This release currently centers on one large Python script plus metadata and packaging files:

- `Beam_weaver.py` — main research code
- `README.md` — project overview and usage notes
- `LICENSE` — software license
- `LICENSE-THIRD-PARTY.md` — third-party boundaries and attribution notes
- `CITATION.cff` — machine-readable citation metadata
- `.zenodo.json` — Zenodo release metadata
- `requirements.txt` / `environment.yml` — environment specifications
- `data/README.md` — notes on expected input data


## INPUT DATA, BINNING, AND GENERATED ARTIFACTS

Beam Weaver uses a small set of physics tables and runtime artifacts.
These should not all be treated in the same way.

========================================================================
CORE TRANSPORT INPUT FILES
========================================================================

The current implementation relies primarily on two photon cross-section
tables:

- Final_cross_sections.csv
  Main photon interaction cross sections as a function of energy.

  This file provides the cross sections used for:
    * photoelectric absorption
    * Compton scattering
    * pair/triplet production

  In the current Beam Weaver implementation, this file also defines the
  main reference energy grid used by the environment. In practice, this
  means that Final_cross_sections.csv determines the principal binning
  used for transport statistics, learning targets, and interaction
  probability calculations. 

- Rayleigh_cross_sections.csv
  Coherent (Rayleigh) scattering cross sections as a function of energy.

  Rayleigh scattering is intentionally stored in a separate input table. This happens because the implementation relied on the PENELOPE cross section tables for water which provides this data separetely as well as with separate binning.
  So, in the current code, it is loaded independently from the other photon
  interaction channels.

  Although Rayleigh data are provided separately, the Beam Weaver
  environment defines its main working energy bins from
  Final_cross_sections.csv. As a result, the Rayleigh contribution is
  effectively mapped onto that binning through interpolation when
  interaction probabilities, attenuation quantities, and related
  transport observables are evaluated.


The following files are also used by the present code.

- OxygenPhotoShells.csv 

   This is the probability interactions for the K, L1 and L2 lines in oxygen as a surrogate for water.


- ElectronStoppingPower.csv

  Auxiliary stopping-power data used for secondary electron transport
  and dose-related calculations. This is only used if the option to explicitely simulate the secondary electrons using a condensed history based code iis used.



# generated running artifacts during pretraining with physics head

The following files are not primary physics input tables. They are
generated during pretraining.

- mc_physics_data.npz
  Monte Carlo-generated dataset used for physics-head pretraining.

- physics_head_pretrained.pth
  Saved checkpoint for the pretrained physics head.
  

## generated running artifacts during training 


- hybrid_sac_model.zip
  Saved Beam Weaver model checkpoint.

- replay_buffer.pkl
  Replay-buffer state used to train the model.


## Legacy/deprecated file not in use


- NIST_WaterCrossSections.csv

  This filename appears in parts of the current script for historical
  reasons, including startup checks and older code paths. However, it
 DOES NOT represent the main transport data source in the current
  implementation.

  It should therefore be treated as a legacy reference rather than as a
  primary required input.
  It can be an empty file

  This legacy dependency is expected to be deprecated and purged in
  future versions of Beam Weaver.


## Notes on binning and consistency


The two main transport files do not need to use identical row-by-row
energy bins in their raw tabulated form. They are loaded independently
and evaluated through interpolation. However:

- Final_cross_sections.csv defines the main energy grid used by the
  current environment. For optimal purposes we suggest to use the binning used by the PENELOPE tabled data

- Rayleigh_cross_sections.csv is expected to cover the same physical
  energy range, even if its native tabulation differs.

- Because the Beam Weaver environment is built on the
  Final_cross_sections.csv energy grid, Rayleigh contributions are
  effectively brought onto that working grid during calculation.

- All physics tables should remain internally consistent in units (MeV, cm2/g),
  ordering, and energy coverage.

----------------------------------------------------------

The script also writes runtime artifacts such as checkpoints, TensorBoard logs, replay-buffer backups, statistics files, and evaluation outputs.

## Quick start

### 1. Create an environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate beam-weaver
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Place the data files

Copy the required CSV, NPZ, PKL, and PTH files into the repository root, or edit the paths in `Rad_sac_24_maio_fixed.py` to match your own layout.

### 3. Run Beam Weaver

```bash
python Rad_sac_24_maio_fixed.py
```

The script is interactive and can be used to:

- generate Monte Carlo data for pretraining
- pretrain the physics head
- train a new SAC model
- evaluate an existing trained model
- compare MC and agent outputs

## Typical workflow

### Generate Monte Carlo pretraining data
Run the script and generate the Monte Carlo dataset used for physics-head supervision.

Expected output:
- `mc_physics_data.npz`

### Pretrain the physics head
Run the pretraining path to fit the physics-informed subnetwork on Monte Carlo-derived targets.

Expected output:
- `physics_head_pretrained.pth`

### Train a policy
Choose training mode and provide total timesteps and an evaluation energy.

Typical outputs include:
- `hybrid_sac_model.zip`
- `replay_buffer.pkl`
- `histograms.pkl`
- `acc_stats.pkl`
- `dist_stats.pkl`
- `tb_logs/`
- `older/`

### Evaluate a trained policy
Choose evaluation mode and provide a fixed photon energy.

The current evaluation path expects:
- `hybrid_sac_model.zip`
- `replay_buffer.pkl`

## Current limitations

This public release should be understood as a research snapshot. In particular:

- the codebase is currently monolithic rather than modular
- execution is interactive rather than command-line driven
- file paths are still largely root-relative / hard-coded
- the geometry and material scope are intentionally narrow
- the angular component still requires further fine-tuning
- validation remains proof-of-concept rather than production-grade
- redistribution of some external data tables may depend on their original terms

## Authors and affiliation

**Pedro Teles**  
Assistant Professor  
Department of Physics and Astronomy  
Faculty of Sciences  
University of Porto

**João Melo**  
MSc Student in Medical Physics  
Department of Physics and Astronomy  
Faculty of Sciences  
University of Porto

## References and acknowledgment

Beam Weaver was developed as an independent Python research implementation, but the in-built Monte Carlo simulator and the surrounding transport logic were developed with substantial reference to the literature and, in particular, to **PENELOPE**. That influence should be acknowledged explicitly in any academic or public presentation of this repository.

In particular, the project should clearly acknowledge that the Monte Carlo side of Beam Weaver was developed with significant conceptual and practical guidance from:

1. **Francesc Salvat.** *PENELOPE-2018: A Code System for Monte Carlo Simulation of Electron and Photon Transport.* OECD Nuclear Energy Agency, NEA/MBDAV/R(2019)1, 2019.
2. **Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.** *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.* ICML, 2018.
3. **Antonin Raffin et al.** *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* Journal of Machine Learning Research, 22:1–8, 2021.


Beam Weaver does **not** distribute the official PENELOPE source code unless a user adds it separately under its own terms. However, the debt to PENELOPE as a scientific and methodological reference is real and should be stated explicitly.

## Citation

Beam Weaver should be cited through the archived Zenodo release for the specific version used. The software record should list both authors.

A machine-readable citation file is provided in `CITATION.cff`, and Zenodo metadata is provided in `.zenodo.json`.

## License

Beam Weaver’s original code in this repository is released under the **Apache License 2.0**.

Third-party data, manuals, and dependencies remain under their own respective licenses and usage terms. See `LICENSE-THIRD-PARTY.md`.
