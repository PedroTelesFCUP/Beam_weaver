# Beam Weaver

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18994135.svg)](https://doi.org/10.5281/zenodo.18994135)


**Physics-informed n-step reinforcement learning for stochastic photon transport in water**

> Beam Weaver is a research software prototype for learning event-by-event photon transport in a homogeneous water phantom. It combines a custom Monte Carlo reference simulator, a physics-informed pretrainable, unconventional n-step Soft-Actor-Critic (discrete and continuous) reinforcement learning environment, and evaluation tools that compare learned particle histories against conventional transport outputs such as interaction statistics, secondary production, particle tracks, and percentage depth dose (PDD).

**Status:** research prototype  
**License:** Apache-2.0  
**Primary domain:** radiation transport and AI research

**Authors:**  
- **Pedro Teles** — Assistant Professor, Department of Physics and Astronomy, Faculty of Sciences, University of Porto  
- **João Melo** — MSc Student in Medical Physics, Department of Physics and Astronomy, Faculty of Sciences, University of Porto

## Research-use notice

Beam Weaver is released as open-source research software under the Apache-2.0 license.
It is currently intended for research and development purposes only and is not validated
for clinical, safety-critical, or production use.

## What Beam Weaver does

Beam Weaver explores whether a **Soft Actor–Critic (SAC)** agent can learn the stochastic logic of photon transport in water at the **event level**, rather than only predicting final observables.

Instead of treating transport as a black-box regression problem, Beam Weaver models the transport process step by step. For each photon history, the code is designed to predict quantities such as:

- the next interaction type
- the distance to the next interaction
- scattering geometry
- the outgoing photon state
- the number, energy, and direction of secondary particles

This makes Beam Weaver closer to a **learned transport-engine prototype** than to a generic machine-learning regressor.


### N-step bootstrapping (non-standard SAC extension)
Beam Weaver does **not** use a strictly vanilla 1-step SAC update. Instead, the current implementation uses an **n-step return** in the critic target through a custom replay-buffer pathway.

This matters because n-step bootstrapping can propagate delayed reward information more quickly than a purely 1-step target, while still retaining a bootstrap term instead of waiting for a full Monte Carlo return. In practice, this gives a useful bias-variance compromise: compared with 1-step updates, multi-step targets can improve credit assignment and sample efficiency; compared with very long returns, they can remain more stable and practical for off-policy learning.

In Beam Weaver, this choice was motivated by the long-horizon, sequential nature of event-by-event transport. A photon history is not a single isolated action but a chain of physically coupled interaction decisions (like Compton showers), so a multi-step target can help the critic connect local decisions to downstream transport consequences more effectively than a purely 1-step backup.

Readers should therefore understand Beam Weaver as a **hybrid SAC variant with n-step bootstrapping**, rather than as a direct drop-in implementation of the original SAC algorithm.

## Scientific scope

The current implementation studies photon transport in a **homogeneous water phantom** and packages a single main research script, `Beam_weaver.py`, which contains:

- a custom Monte Carlo-style photon transport simulator in water
- a custom `WaterPhotonHybridEnvPenelope` environment
- a hybrid n-step SAC actor-critic implementation for mixed discrete/continuous prediction
- curriculum-based training logic with phase switching
- physics-head pretraining on Monte Carlo-generated targets
- evaluation utilities for comparing Monte Carlo and agent-generated showers

In its current form, Beam Weaver models all four photon interaction channels in water:

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

Monte Carlo simulation transport remains the reference approach for detailed stochastic transport simulation, but it is often computationally expensive. Beam Weaver was created as a proof-of-concept framework for studying whether reinforcement learning can reproduce key parts of the photon transport loop while remaining physically grounded.

The project therefore sits at the intersection of:

- iation transport simulation
- reinforcement learning
- physics-informed machine learning
- surrogate modeling for Monte Carlo workflows

## What Beam Weaver does **not** claim

Beam Weaver is **not** a validated Monte Carlo engine, or a completed production codebase.


Its purpose is research: to investigate whether a physics-informed RL agent can learn the structure of stochastic photon transport from a Monte Carlo reference process.

## Learning architecture and training workflow

Beam Weaver is trained in two distinct stages: (i) simulation-supervised
pretraining of a physics branch, and (ii) curriculum-based off-policy
reinforcement learning with staged teacher forcing.

### Stage I — pre-training: supervised learning of the physics head on Monte Carlo-generated labels**

Before SAC optimization begins, Beam Weaver generates a Monte
Carlo dataset of event-level transport targets. For each sampled
interaction, the code stores an observation vector together with:
- free path,
- interaction "class",
- photon scattering angle,
- outgoing photon energy,
- number of secondaries,
- Secondary-particle energies and angles,


The observation vector is z-scored before pretraining. The physical
targets are also transformed into numerically convenient forms:
- The interaction class is kept as
  discrete labels,
- free path and energies are mapped as `log(1 + x)`,
- angular quantities are represented as `sin(theta)` and `cos(theta)`,
- The number of secondaries, also kept as a discrete label.


The physics branch is not a single scalar regressor. It is a structured
multi-head module attached to the shared feature extractor. In the
current implementation:
- the shared feature extractor maps the observation to a latent feature
  vector;
- a physics backbone processes that latent representation;
- a process head predicts the interaction class;
- an energy head predicts interaction-conditional energy quantities;
- an angle head predicts interaction-conditional angular quantities;
- a secondary-count head predicts the number of secondaries.

The energy and angle heads are conditioned not only on the latent
physics representation, but also on:
- the predicted interaction probabilities, and
- a one-hot encoding of the current energy bin.

This means the physics branch is trained as a conditional predictor of
interaction-specific transport kernels rather than as a single global
regressor.

The pretraining loss is a weighted sum of four terms:

$$
\mathcal{L}_{\mathrm{pre}} = w_E \mathcal{L}_E + w_\theta \mathcal{L}_\theta + w_N \mathcal{L}_N +w_P \mathcal{L}_P, 
$$

with equal default weights

$$
w_E = w_\theta = w_N = w_P = 0.25.
$$

In the current code:
- $\mathcal{L}_P$ is a cross-entropy loss for the interaction type;
- $\mathcal{L}_E$ is a masked mean-squared loss over `log(1+fp)`;
  `log(1+E_out)`, and the secondary log-energies;
- $\mathcal{L}_\theta$ is a masked mean-squared loss over the sine/cosine
  representation of photon and secondary angles;
- $\mathcal{L}_N$ is a cross-entropy loss for the number of secondaries.

Secondary particle production is masked so that nonexistent secondaries do not
contribute to the objective, this means for instance that Photoelectric or Rayleigh interactions
will inherently not produce secondaris. A small Gaussian perturbation is also added
to normalized observations during pretraining for robustness. The
optimizer is Adam with weight decay, and validation loss is monitored
with a ReduceLROnPlateau scheduler.

This stage can therefore be described as **supervised learning on
Monte Carlo-generated labels**.

### Stage II — curriculum-based off-policy SAC with staged teacher forcing

After pretraining, Beam Weaver enters a curriculum-based hybrid SAC
training stage. The actor has three functional parts:
1. a discrete policy head for interaction selection;
2. a continuous policy head for transport parameters;
3. the pretrained physics branch, which is reused as an auxiliary
   physics regularizer during RL training.

The observation space encodes:
- normalized photon position,
- normalized photon energy and log-energy,
- photon direction cosines,
- local transport-step descriptors,
- normalized interaction cross sections,
- a fixed-size secondary-history buffer,
- and a shell one-hot context vector.

The action space is hybrid but flattened for implementation:
- the first component is a discrete interaction choice
  (`rayleigh`, `compton`, `photoelectric`, `pair`);
- the remaining components parameterize the remainder continuous outputs such as free
  path / attenuation behavior, scattered-photon quantities, and
  secondary-particle energies and directions.

### Curriculum phases

Beam Weaver does not train all policy components from the 
beginning or at the same time. Instead, it uses staged curriculum learning.

#### Phase 0 — discrete interaction learning, fully teacher-forced
Only the shared feature extractor and the discrete head are trainable.
The discrete interaction choice is overridden by the Monte Carlo truth.
At the start of this phase, the actor also initializes:
- the discrete logits from physics-derived interaction probabilities,
- and the attenuation / mean-free-path component from the tabulated
  physics reference.

The mean-free-path residual output and its corresponding Gaussian
variance are then frozen.

#### Phase 1 — discrete interaction learning, teacher forcing decays out
The same trainable subset is kept, but the discrete teacher forcing is
progressively reduced. In the current implementation this decay is
linear over a fixed horizon. This stage therefore acts as scheduled
teacher forcing from Monte Carlo-labeled interaction selection toward
autonomous discrete policy learning.

#### Phase 2 — continuous/kernel learning, fully teacher-forced
The discrete head is frozen. The rest of the continuous/kernel branches
remain trainable, while the mean-free-path output stays fixed to its
physics initialization. During this phase, the Monte Carlo kernel values
are forced into the stored action representation so that the replay
buffer contains Monte Carlo-consistent targets for continuous transport
quantities.

#### Phase 3 — continuous/kernel learning, teacher forcing decays out
This phase keeps the same branch structure as Phase 2, but gradually
reduces the kernel-level teacher forcing. In other words, Beam Weaver
transitions from MC-driven kernel supervision to autonomous continuous
prediction.

#### Phase 4 and beyond — hybrid refinement with frozen MFP output
The discrete head remains frozen, the mean-free-path output remains
frozen, and the remaining continuous transport heads (energy loss, angles, secondary particle energy) 
continue to be
optimized.

### Replay buffer and off-policy supervision

Beam Weaver uses a custom n-step replay buffer. This is important for
two reasons.

First, the TD target uses an n-step return rather than a purely 1-step
backup:


$$
y_t^{(n)} = r_t^{(n)} + \gamma^n \left(Q_{\mathrm{target}}(s_{t+n}, a_{t+n}) - \alpha \log \pi(a_{t+n}\mid s_{t+n})\right).
$$


Second, during teacher-forced phases, the replay buffer stores the
overridden Monte Carlo-consistent action rather than the raw action
proposed by the agent. This is a crucial design choice: the off-policy
dataset seen by the critic and by the auxiliary physics losses is
therefore anchored to the Monte Carlo reference during curriculum
stages, and only later becomes fully policy-driven as teacher forcing
is withdrawn.

### RL objective during SAC training

During RL, Beam Weaver does not optimize a pure SAC objective alone.
Instead, the actor loss is multitask:


$$
\mathcal{L}_{\mathrm{actor}}=\mathcal{L}_{\mathrm{SAC}}+\lambda_{\mathrm{phys}}\mathcal{L}_{\mathrm{phys}},
$$


where

$$
\mathcal{L}_{\mathrm{SAC}}=\mathbb{E}\left[\alpha \log \pi(a\mid s) - Q(s,a)\right].
$$

The auxiliary physics term currently takes the form

$$
\mathcal{L}_{\mathrm{phys}}=\mathcal{L}_E+\mathcal{L}_\theta+0.4  \mathcal{L}_{\mathrm{norm}}+\mathcal{L}_N+\mathcal{L}_P,
$$

where:
- $\mathcal{L}_E$ is the interaction-aware energy loss,
- $\mathcal{L}_\theta$ is the interaction-aware angular loss,
- $\mathcal{L}_{\mathrm{norm}}$ penalizes sine/cosine pairs that drift
  away from unit norm,
- $\mathcal{L}_N$ is the number-of-secondaries classification loss,
- $\mathcal{L}_P$ is the interaction-type classification loss.

Thus, Beam Weaver is best understood as an n-step **physics-regularized hybrid
SAC framework** rather than a vanilla SAC implementation.

### Reward design and curriculum targets

Beam Weaver does not use a single stationary reward throughout training.
Instead, the reward is **phase-dependent** and is switched explicitly by
the curriculum wrapper.

Let the full environment-level reward be

$$
R_t^{\mathrm{full}} = r_{\mathrm{disc},t} + r_{\mathrm{kernel},t} + r_{E\text{-corr},t}.
$$

In the current implementation, the curriculum wrapper does **not**
always expose this full reward to SAC. Instead, it returns:

- **Phase 0:**

$$
R_t^{(0)} = r_{\mathrm{disc},t}
$$

- **Phase 1:**

$$
R_t^{(1)} = r_{\mathrm{disc},t}
$$

- **Phase 2 and later curriculum phases:**

$$
R_t^{(\ge 2)} = r_{\mathrm{kernel},t} + r_{E\text{-corr},t}
$$
  
  

So the early curriculum is a discrete-policy learning stage, whereas
the later curriculum is a continuous/kernel learning stage.

#### 1. Discrete interaction reward used in phases 0--1

In phases 0--1, the goal is to make the agent reproduce the correct
interaction mixture

$$
\pi_{\mathrm{true}}(a\mid E)
$$

on the Beam Weaver energy grid.

At each step, the incoming photon energy is assigned to an energy bin
$b(E_t)$. The code maintains:

- an exponentially weighted moving histogram $H_{b,a}$ of recent
  agent-selected discrete actions;
- a cumulative histogram $C_{b,a}$ used for diagnostics and bin
  validity checks.

The EWMA update is

$$
H_{b,a} \leftarrow \lambda_{\mathrm{hist}} H_{b,a}
$$

followed by

$$
H_{b(E_t),a_t} \leftarrow H_{b(E_t),a_t} + 1,
$$

with

$$
\lambda_{\mathrm{hist}} = 0.995.
$$


Only bins with at least 10 cumulative events are used in the global
mixture estimate. Let

$$
w_b = \mathbf{1}\!\left[\sum_a C_{b,a} \ge 10\right]\sum_a H_{b,a}.
$$

The weighted target mixture is then

$$
p_{\mathrm{true},a} = \frac{\sum_b w_b   \pi_{\mathrm{true},b,a}}{\sum_b w_b},
$$

where $\pi_{\mathrm{true},b,a}$ is the tabulated interaction
probability in energy bin $b$.

The discrete reward for the chosen action $a_t$ is

$$
r_{\mathrm{disc},t}=\log\!\left(\frac{p_{\mathrm{true},a_t}+\varepsilon}{1/4+\varepsilon}\right),
$$

with a uniform baseline $1/4$ and small numerical $\varepsilon$.

This is therefore a **log-likelihood-ratio reward** that favors
interaction choices that are more probable under the physics mixture
than under a uniform baseline.

For monitoring only, the code also computes:
- the Jensen--Shannon divergence between the recent action histogram and
  the target mixture,
- and the global $L^1$ distance between them.

These quantities are logged, but they are not directly added to the
reward.

#### 2. Kernel reward used from phase 2 onward

Once the discrete curriculum is complete, the reward shifts to a
continuous/kernel objective. The general form is

$$
r_{\mathrm{kernel},t}=r_{\mathrm{dist},t}+r_{\phi,t}+r_{\phi_e,t}+r_{\theta,\mathrm{pair},t}+r_{E,\mathrm{pair},t}+r_{e,\mathrm{comp},t}.
$$

This total is clipped to

$$
r_{\mathrm{kernel},t} \in [-50,50].
$$

The terms are process-specific.

##### 2.1 Distribution-matching term

For all four interactions, the code calls a process-specific acceptance
kernel

$$
A_i(\cdot)
$$

through `accept_prob(...)`, where $i\in\{\text{Rayleigh, Compton,
Photoelectric, Pair}\}$. The primary angular reward is

$$
r_{\mathrm{dist},t}^{(i)}=\log\!\left(\frac{A_i+\varepsilon}{1/180+\varepsilon}\right),
$$

where the baseline $1/180$ reflects the histogram discretization in
degrees.

This term says: if the predicted angle lands in a region of high
physical kernel density, reward it; if it lands in a region that is only
as good as a uniform angle, reward is small.

##### 2.2 Per-bin KL angular-distribution penalty

For each energy bin and interaction type, Beam Weaver also keeps a short
history of generated angles and compares the empirical histogram against
the target angular distribution returned by `accept_prob(...)`.

If there are at least 20 stored angles in the corresponding
energy--interaction bin, the code computes

$$
D_{\mathrm{KL}}(p_{\mathrm{target}}\|p_{\mathrm{empirical}}) = \sum_j p_{\mathrm{target},j} \log\!\left( \frac{p_{\mathrm{target},j}}{p_{\mathrm{empirical},j}} \right),
$$




and adds the penalty

$$
r_{\mathrm{KL},t}^{(i)}=-0.05  \mathrm{clip}\!\left(D_{\mathrm{KL}}  ,0,  10\right).
$$

So the actual distribution term is

$$
r_{\mathrm{dist},t}^{(i)}\leftarrow r_{\mathrm{dist},t}^{(i)} + r_{\mathrm{KL},t}^{(i)}.
$$

##### 2.3 Rayleigh-specific energy consistency

For Rayleigh scattering, the outgoing photon energy should equal the
incoming photon energy. The code adds

$$
r_{E,\mathrm{Rayleigh},t}=-0.01  \frac{|E_{\gamma}^{\mathrm{pred}}-E_{\gamma}^{\mathrm{in}}|}{\max(E_{\gamma}^{\mathrm{in}},10^{-6})}.
$$

So the Rayleigh kernel reward is

$$
r_{\mathrm{kernel},t}^{\mathrm{Rayleigh}}=r_{\mathrm{dist},t}^{\mathrm{Rayleigh}}+r_{E,\mathrm{Rayleigh},t}+r_{\phi,t}.
$$

##### 2.4 Compton-specific electron kinematic consistency

For Compton scattering, the code derives the recoil-electron reference
kinematics from the predicted photon branch:

$$
E_{e,\mathrm{MC}} = E_{\gamma}^{\mathrm{in}} - E_{\gamma}^{\mathrm{pred}},
$$

$$
p_{e,\mathrm{MC}} =\sqrt{E_{e,\mathrm{MC}}(E_{e,\mathrm{MC}}+2m_ec^2)},
$$

$$
\cos\theta_{e,\mathrm{MC}}=\frac{E_{\gamma}^{\mathrm{in}}-E_{\gamma}^{\mathrm{pred}}\cos\theta_{\gamma}^{\mathrm{pred}}}
{\max(p_{e,\mathrm{MC}},10^{-12})}.
$$

The electron consistency penalty is then

$$
r_{e,\mathrm{comp},t}=-\left[0.05  \frac{|E_{e}^{\mathrm{pred}}-E_{e,\mathrm{MC}}|}{\max(E_{e,\mathrm{MC}},10^{-6})}+0.05  \frac{|\theta_{e}^{\mathrm{pred}}-\theta_{e,\mathrm{MC}}|}{\pi}\right].
$$

So the Compton kernel reward is
$$
r_{\mathrm{kernel},t}^{\mathrm{Compton}}=r_{\mathrm{dist},t}^{\mathrm{Compton}}+r_{e,\mathrm{comp},t}+r_{\phi,t}+r_{\phi_e,t}.
$$

##### 2.5 Photoelectric reward

For photoelectric events, the angular term is evaluated on the emitted
electron angle using the shell-specific acceptance model:

$$
r_{\mathrm{dist},t}^{\mathrm{Photo}}=\log\!\left(\frac{A_{\mathrm{photo}}+\varepsilon}{1/180+\varepsilon}\right)+r_{\mathrm{KL},t}^{\mathrm{Photo}}.
$$

The code also computes a photoelectric electron-energy penalty

$$
r_{E,\mathrm{Photo},t}=-0.01  \frac{|E_{e}^{\mathrm{pred}}-E_{\mathrm{avail}}|}{\max(E_{\mathrm{avail}},10^{-6})},
$$

but in the current implementation this term is computed but not added
to $r_{\mathrm{kernel}}$.


##### 2.6 Pair-production reward

For pair production, the code combines:
- angular distribution matching,
- electron azimuthal-uniformity regularization,
- pair energy-sharing consistency,
- and a small-angle consistency term.

The pair energy-sharing penalty is

$$
r_{E,\mathrm{pair},t}=-0.05  \frac{|E_e^{\mathrm{pred}}+E_p^{\mathrm{pred}}-(E_{\gamma}^{\mathrm{in}}-2m_ec^2)|}{E_{\gamma}^{\mathrm{in}}-2m_ec^2}.
$$

The polar-angle penalty is

$$
r_{\theta,\mathrm{pair},t}=-\lambda_{\theta}\frac{|\theta_e^{\mathrm{pred}}-\theta_{\mathrm{MC}}|}{\theta_{\mathrm{MC}}},
$$

where $\theta_{\mathrm{MC}}$ is a reference small-angle model.

So the intended pair kernel reward is

$$
r_{\mathrm{kernel},t}^{\mathrm{Pair}}=r_{\mathrm{dist},t}^{\mathrm{Pair}}+r_{E,\mathrm{pair},t}+r_{\theta,\mathrm{pair},t}+r_{\phi_e,t}.
$$

The code also computes a positron azimuthal-uniformity term
$r_{\phi_p}$, but this term is currently not added to
$r_{\mathrm{kernel}}$.

#### 3. Azimuthal-uniformity terms

For processes where azimuth should be uniform, Beam Weaver uses the
first circular moment:

$$
R = \left|\frac{1}{N}\sum_{n=1}^{N} e^{i\phi_n}\right|.
$$

The corresponding penalty is

$$
r_{\phi} = -\lambda_{\phi} R,\qquad\lambda_{\phi}=0.05.
$$

This is applied:
- to photon azimuth for Rayleigh and Compton,
- to electron azimuth for Compton, photoelectric, and pair production.

The code uses batches of 32 accumulated samples before applying the
penalty.

#### 4. Explicit forbidden-photon-energy penalty for photoelectric and pair

For photoelectric absorption and pair production, the outgoing photon
energy should be zero. The code therefore adds

$$
r_{E\text{-corr},t}=\begin{cases}-5, & \text{if } a_t\in\{\text{photo},\text{pair}\}\text{ and } E_{\gamma}^{\mathrm{pred}}>10^{-12},\\0,  &\text{otherwise.}\end{cases}
$$

This term is always added during the kernel-learning stages because the
wrapper returns
$$
R_t^{(\ge 2)} = r_{\mathrm{kernel},t} + r_{E\text{-corr},t}.
$$

#### 5. Teacher forcing and reward exposure by phase

The reward phases are aligned with the teacher-forcing schedule:

- **Phase 0:** full teacher forcing of the discrete interaction; reward
  exposed to SAC is only $r_{\mathrm{disc}}$.
- **Phase 1:** discrete teacher forcing is linearly decayed; reward
  exposed to SAC is still only $r_{\mathrm{disc}}$.
- **Phase 2:** kernel outputs are fully overwritten by MC values in the
  stored action; reward exposed to SAC is
  $r_{\mathrm{kernel}} + r_{E\text{-corr}}$.
- **Phase 3:** kernel teacher forcing is linearly decayed; reward
  remains $r_{\mathrm{kernel}} + r_{E\text{-corr}}$.
- **Later phases:** the code continues using the kernel-stage reward
  structure.

In other words, the curriculum does not merely freeze and unfreeze
different heads. It also changes which part of the objective the agent
actually sees.

### Entropy control

Beam Weaver keeps SAC’s entropy-regularization mechanism and learns the
entropy coefficient automatically. The code tracks the entropy
coefficient, discrete entropy, and continuous entropy throughout
training. This matters because exploration requirements differ strongly
between the discrete curriculum stages and the later kernel-learning
stages.

In summary, Beam Weaver should not be described as a single homogeneous
RL loop. It is a staged, simulation-supervised + off-policy RL training
pipeline in which Monte Carlo targets are first learned directly and
then progressively relinquished as the policy takes control.

## Repository contents

This release currently centers on one large Python script plus metadata and packaging files:

- `Beam_weaver_0.1.0.py` — main research code
- `README.md` — project overview and usage notes
- `LICENSE` — software license
- `LICENSE-THIRD-PARTY.md` — third-party boundaries and attribution notes
- `CITATION.cff` — machine-readable citation metadata
- `.zenodo.json` — Zenodo release metadata
- `requirements.txt` / `environment.yml` — environment specifications
- `data/README.md` — notes on expected input data


## Input data, binning, and generated artifacts

Beam Weaver uses a small set of physics tables and runtime artifacts.
These should not all be treated in the same way.

### Core transport input files

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

- water_sq.csv
  Incoherent scattering function from Hubbel et al (1975)
using the independent-atom,
stoichiometric sum for water:


`S_H2O(x) = 2*S_H(x) + S_O(x)`


where `x = sin(theta/2)/lambda`.

For compatibility with the current Beam Weaver code, the file is stored with
columns `q,S_q`, although the tabulated axis corresponds to Hubbell's
`x = sin(theta/2)/lambda` rather than the full scattering-vector magnitude.
This matches the current implementation, which computes the full momentum
transfer and then converts it to the table variable internally through
`x_table = q / 2`.

Note that the coherent form Factor is built from a Cromer-Mann style interpolation.


The following files are also used by the present code.

- OxygenPhotoShells.csv 

   This is the probability interactions for the K, L1 and L2 lines in oxygen as a surrogate for water.


- ElectronStoppingPower.csv

  Auxiliary stopping-power data used for secondary electron transport
  and dose-related calculations. This is only used if the option to explicitely simulate the secondary electrons using a condensed history based code iis used.



### generated running artifacts during pretraining with physics head

The following files are not primary physics input tables. They are
generated during pretraining.

- mc_physics_data.npz
  Monte Carlo-generated dataset used for physics-head pretraining.

- physics_head_pretrained.pth
  Saved checkpoint for the pretrained physics head.
  

### generated running artifacts during training 


- hybrid_sac_model.zip
  Saved Beam Weaver model checkpoint.

- replay_buffer.pkl
  Replay-buffer state used to train the model.


### Legacy/deprecated file not in use


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


### Notes on binning and consistency


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

We strongly recommend running this software on a GPU available architecture.

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

Copy the required CSV, NPZ, PKL, and PTH files into the repository root, or edit the paths in `Beam_weaver_0.1.0.py` to match your own layout.

### 3. Run Beam Weaver

```bash
python Beam_weaver_0.1.0.py
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

## Current project status

Beam Weaver's central purpose of the project was largely achieved: the code can already simulate the main photon interaction processes in a physically meaningful way and support comparison between the learned agent and the Monte Carlo reference workflow. The main remaining development task  is  **fine-tuning the angular predictions / angular sampling behavior**, which is the principal unfinished component at this time.

In other words, this project is a working research prototype with a specific remaining refinement target.

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

Beam Weaver was developed as an independent Python research implementation, but the in-built Monte Carlo simulator and the surrounding transport logic were developed with substantial reference to the literature and, in particular, to **PENELOPE**. 

In particular, the project clearly acknowledges that the Monte Carlo side of Beam Weaver was developed with significant conceptual and practical guidance from:

1. **Francesc Salvat.** *PENELOPE-2018: A Code System for Monte Carlo Simulation of Electron and Photon Transport.* OECD Nuclear Energy Agency, NEA/MBDAV/R(2019)1, 2019.
2. **Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.** *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.* ICML, 2018.
3. **Antonin Raffin et al.** *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* Journal of Machine Learning Research, 22:1–8, 2021.
4. **J. H. Hubbell et al.?** *Atomic Form Factors, Incoherent Scattering Functions, and Photon Scattering Cross Sections*. Journal of Physical and Chemical Reference Data, 4(3), 471--538. 1975. https://doi.org/10.1063/1.555523



## Citation

Beam Weaver should be cited through the archived Zenodo release for the specific version used. The software record should list both authors.

A machine-readable citation file is provided in `CITATION.cff`, and Zenodo metadata is provided in `.zenodo.json`.

## License

Beam Weaver’s original code in this repository is released under the **Apache License 2.0**.

Third-party data, manuals, and dependencies remain under their own respective licenses and usage terms. See `LICENSE-THIRD-PARTY.md`.
