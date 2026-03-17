---
title: 'Beam Weaver - an unconventional n-step Soft-Actor Critic agent for photon transport in water'
tags:
  - Python
  - reinforcement learning
  - Soft Actor-Critic
  - Monte Carlo
  - radiation transport
  - medical physics
  - scientific machine learning
authors:
  - name: Pedro Teles
    corresponding: true
    orcid: 0000-0003-2705-7077
    affiliation: 1
  - name: João Melo
    affiliation: 1
affiliations:
  - name: Department of Physics and Astronomy, Faculty of Sciences, University of Porto, Portugal
    index: 1
date: 13 March 2026
bibliography: paper.bib
---

# Summary

Beam Weaver is an open-source Python research framework for learning
event-by-event photon transport in a homogeneous water phantom using a
physics-informed hybrid Soft Actor-Critic (SAC) agent. The software
combines three tightly coupled components: (i) an internal Monte Carlo
(MC) reference simulator that samples photon interactions from
PENELOPE-derived cross-section tables [@penelope2018], (ii) a
Gymnasium-compatible reinforcement-learning environment that casts
photon transport as a structured sequential decision problem
[@brockman2016openai], and (iii) a custom multi-head SAC
implementation that jointly predicts a discrete interaction type and
continuous transport quantities including free path, scattering
geometry, outgoing photon energy, and secondary-particle properties.

In contrast to standard SAC implementations [@haarnoja2018sac], Beam
Weaver uses n-step bootstrapped returns in the critic target
[@suttonbarto2018] and a five-phase curriculum that stages training
from fully teacher-forced discrete interaction learning through
progressively autonomous continuous transport prediction. The current
release models the four dominant photon interactions in
water---Rayleigh scattering, Compton scattering, photoelectric
absorption, and pair production---together with an optional simplified
condensed-history secondary-electron transport mode for depth-dose
estimation, or alternatively a local energy deposition approach.

Beam Weaver is released as a transparent, reproducible research
platform for studying whether reinforcement learning can capture the
stochastic event structure of radiation transport while retaining
physically interpretable behaviour. The software is functional in its
target water-phantom setting, although it remains a work in progress:
core interaction simulation is operational, while angular distribution
fine-tuning is the principal unfinished component. Releasing Beam
Weaver in this form makes its unusual architecture, staged training
workflow, and evaluation pipeline available to the community for
scrutiny, reuse, and collaboration.

# Statement of need

High-fidelity radiation transport is typically performed with mature
Monte Carlo codes such as PENELOPE, Geant4, and EGSnrc
[@penelope2018; @agostinelli2003geant4; @kawrakow2000egsnrc]. These
remain the standard for validated transport and dosimetric accuracy,
but their computational cost can limit rapid experimentation and
repeated evaluation in algorithm-development settings. At the same
time, reinforcement-learning libraries such as Stable-Baselines3
[@raffin2021stable] provide robust algorithmic infrastructure but do
not include radiation-physics environments, transport samplers, or
validation workflows tailored to stochastic photon transport.

The existing machine-learning literature on accelerating radiation
transport has focused predominantly on surrogate models for final
observables---dose maps, depth-dose curves, or denoising of low-photon
Monte Carlo outputs---rather than on learning the event-level
transport process itself. 

Beam Weaver addresses a different question:
can a reinforcement-learning agent learn to reproduce the stochastic
sequence of interaction decisions that constitutes a photon history,
not merely predict downstream aggregate quantities?

Beam Weaver was developed for researchers in medical physics,
computational radiation transport, and scientific machine learning who
want an open platform for investigating learned transport surrogates at
the event level. Rather than learning a final dose map directly, Beam
Weaver models the transport process itself: the agent observes photon
state variables and cross-section-derived quantities, then predicts the
next interaction class together with continuous outcomes that can be
compared against MC-generated reference behaviour.

The release described here is also motivated by reproducibility and
community visibility. Beam Weaver bundles the complete workflow needed
to generate MC-derived supervision targets, pretrain a physics branch,
train the hybrid SAC policy through a staged curriculum, save
checkpoints, and compare learned transport behaviour against MC
references through interaction statistics, secondary-particle
summaries, particle tracks, and percentage depth-dose (PDD) curves. It
is therefore not merely a model checkpoint but an inspectable research
pipeline for learned transport. Making that pipeline public at this
stage is valuable even though the project is not yet complete, because
the software already demonstrates a distinctive combination of
structured RL actions, physics-guided supervision, curriculum training,
and n-step SAC updates that is not available in any existing package.

# State of the field

Established transport packages such as PENELOPE, Geant4, and EGSnrc
provide validated and extensible particle-transport capabilities across
broad classes of materials, geometries, and interaction models
[@penelope2018; @agostinelli2003geant4; @kawrakow2000egsnrc] have become the 
pinnacle of particle transport and are used to infer reference dosimetric quantities, 
make detailed dosimetric analysis, study particle tracks and other tasks.

Beam Weaver intends to be a proof-of-concept to address the question, can a well-trained reinforcement-learning 
agent transport photons as fast and as accurately as a Monte Carlo code?
It should be said that Beam Weaver does not intend to compete with Monte Carlo cdodes
on scope, maturity, or clinical readiness, but rather try to demonstrate the feasibility
of training an agent to transport simulation which perhaps, in the future, can replace or assist 
Monte Carlo codes to increase calculation speed and accuracy. 
The Soft Actor-Critic (SAC) architecture was chosen for several reasons that align with the
structure of stochastic photon transport. First, SAC's maximum-entropy
formulation produces inherently stochastic policies, which is
physically appropriate: at any given photon energy, multiple
interaction types carry non-zero probability and scattering angles
follow broad process-specific distributions, so the correct learned
behaviour is a probability distribution over actions rather than a
deterministic mapping. The entropy-regularization term actively
prevents premature collapse to a single interaction type or a narrow
angular mode, a failure pattern that would be physically incorrect even
if it locally maximized a naive reward signal. Second, SAC is
off-policy: it can learn from a replay buffer that includes
teacher-forced transitions recorded during early curriculum phases,
which is essential for the staged training design in which the buffer
initially contains MC-overridden actions and only later becomes fully
policy-driven. On-policy methods such as Proximal Policy Optimization (PPO) would discard this
supervised data after each update, losing the benefit of anchoring
early learning to the MC reference. Third, SAC handles continuous
action spaces natively, which is required for predicting free path,
scattering angles, energies, and secondary-particle properties in a
single forward pass. Fourth, the automatic entropy-coefficient tuning
in SAC provides a mechanism for adjusting exploration pressure across
curriculum phases without manual scheduling: the discrete phases
require broad categorical exploration across four interaction channels,
while the later kernel phases require finer continuous exploration
around angular and energy targets, and these regimes have very
different natural entropy scales.

Beam weaver uses a deliberately narrow setting to make its point ---a $10 \times 10$ cm$^2$ monoenergetic
photon beam incident on a $100 \times 100 \times 100$ cm$^3$
homogeneous water phantom---to study whether a SAC agent can learn the
event-level structure of radiation transport from a physics-based
simulator.

General-purpose reinforcement-learning frameworks likewise address a
different problem. Stable-Baselines3 and related toolkits provide
training algorithms, policies, buffers, and utilities, but they leave
domain modelling to the user [@raffin2021stable]. They do not include
photon interaction models, particle kinematics, dose tallies, or
transport-specific diagnostics. Standard SAC formulations are also not
designed around the temporally structured reward signals and
credit-assignment patterns that arise when a single photon history
spans a chain of physically coupled interactions
[@haarnoja2018sac].

Beam Weaver was therefore built as a standalone research codebase
rather than as an extension of an existing transport code or a generic
RL package, for three reasons. First, the design goal is
methodological experimentation with learned transport, not an extension
of a production Monte Carlo code. Second, the software needs tight
coupling between MC-generated labels, staged reinforcement-learning
curricula, and transport-specific reward shaping that would be
difficult to retrofit into an existing framework. Third, the code
exposes intermediate physical quantities---interaction statistics,
secondary-particle properties, per-process angular distributions,
particle tracks, and depth-dose curves---that are especially useful for
comparing learned and reference transport process by process.

# Software design

## Overview

Beam Weaver is organized around three tightly coupled components: a
Monte Carlo reference simulator, a Gymnasium-compatible
reinforcement-learning environment, and a custom SAC training stack.
The current release is distributed as a single Python research script,
`Beam_weaver_0.1.0.py`, accompanied by physics input tables,
environment specifications, and metadata for citation and archiving.

## Monte Carlo reference simulator

The first component is a compact MC photon transport simulator written
in Python. The simulator uses cross-section tables derived from the
PENELOPE material-data files for water [@penelope2018] (Rayleigh, Compton and Pair Production), 
 the EPDL library [@cullen1997epdl] for the Photoelectric interaction, and Hubbel data for the incoherent
scattering function and the coherent form factor. 

Cross-section data are organized in two input files.
`Final_cross_sections.csv` provides the photoelectric, Compton, and
pair-production cross sections as a function of energy (in MeV, with
values in cm$^2$/g); this file also defines the principal energy grid
used throughout the environment for binning, learning targets, and
interaction-probability calculations. `Rayleigh_cross_sections.csv`
provides the coherent scattering cross section on a separate energy
grid, reflecting the native PENELOPE tabulation; Rayleigh contributions
are mapped onto the main energy grid through interpolation when
interaction probabilities and attenuation quantities are evaluated.

For Compton scattering, the differential cross section is sampled using
the Klein--Nishina formula [@klein1929] modulated by the incoherent
scattering function $S(x)$, where $x = \sin(\theta/2)/\lambda$. Beam
Weaver uses Hubbell-style Waller--Hartree tabulations for hydrogen and
oxygen [@hubbell1975], combined through the independent-atom
stoichiometric sum for water:

$$S_{\mathrm{H_2O}}(x) = 2\,S_{\mathrm{H}}(x) + S_{\mathrm{O}}(x).$$

This quantity is stored in `water_sq.csv`. For compatibility with the
current code, the file uses the column name `q`, although the tabulated
axis corresponds to Hubbell's $x = \sin(\theta/2)/\lambda$; the
sampler internally converts the full momentum transfer to this table
variable before interpolation.

For Rayleigh scattering, angular sampling uses the coherent
(Rayleigh) form factor $F(q)$ for water, tabulated from Hubbell and
co-workers [@hubbell1975] and stored in `water_fq.csv`. Rayleigh
scattering angles are accepted with a weight proportional to

$$\frac{F(q)^2}{F(0)^2}\;\frac{1+\cos^2\theta}{2},$$

combining a Thomson-like angular factor with the coherent form-factor
modulation. The form factor is interpolated from the tabulated values
at runtime rather than approximated by an analytic fit.

For photoelectric absorption, the code uses shell-specific interaction
probabilities for the K and L sub-shells of oxygen as a surrogate for
water, stored in `WaterPhotoShells.csv`. For secondary charged-particle
handling, the implementation draws methodological inspiration from
PENELOPE and Geant4 condensed-history concepts [@penelope2018;
@agostinelli2003geant4], but also provides a simpler local energy
deposition mode that deposits secondary electron energy at the
interaction site.

## Reinforcement-learning environment

The second component is a Gymnasium-compatible environment
(`WaterPhotonHybridEnvPenelope`) that casts photon transport as a
structured sequential decision problem. Each episode corresponds to one
photon history, beginning when a photon enters the phantom and ending
when the photon is absorbed, escapes the phantom geometry, or falls
below a transport energy cutoff.

**Observation space.** At each transport step, the agent observes a
fixed-length vector encoding: normalized photon position (three
Cartesian coordinates), normalized photon energy and its logarithm,
photon direction cosines, local transport-step descriptors (mean free
path and attenuation-related quantities), normalized interaction cross
sections for all four channels, a fixed-size secondary-particle history
buffer, and a shell one-hot context vector identifying the relevant
photoelectric sub-shell. All continuous observations are z-scored using
statistics computed during MC data generation.

**Action space.** The action space is hybrid but flattened into a single
continuous vector for implementation compatibility with
Stable-Baselines3 [@raffin2021stable]. The first component encodes a
discrete interaction choice among the four channels (Rayleigh, Compton,
photoelectric, pair production). The remaining components parameterize
continuous transport outputs: free path and attenuation behaviour,
scattered-photon energy and polar angle (as $\sin\theta$ and
$\cos\theta$ pairs), scattered-photon azimuthal angle, number of
secondary particles, and secondary-particle energies and directions.

**Diagnostics.** The environment exposes transport-specific diagnostics
including energy-binned interaction-frequency histograms,
per-interaction angular-distribution comparisons, secondary-particle
summaries, and PDD behaviour. These diagnostics are central to Beam
Weaver's purpose: the software is intended not only to optimize a
scalar score but to enable process-by-process comparison between
learned and reference transport.

## SAC training stack

The third component is a custom SAC training stack built on
Stable-Baselines3 [@raffin2021stable] and PyTorch [@paszke2019pytorch].
Beam Weaver extends the standard SAC workflow in several ways that make
the implementation materially different from off-the-shelf SAC
[@haarnoja2018sac].

**Multi-head policy architecture.** The policy uses a shared feature
extractor that feeds three functional sub-networks: (i) a discrete
head for interaction-type selection, (ii) continuous heads for
interaction-conditional transport variables (free path, photon energy,
angles, secondary-particle properties), and (iii) a physics branch
composed of auxiliary prediction heads for physically meaningful
quantities. The physics branch is structured as a conditional
predictor: its energy and angle heads receive not only the latent
physics representation but also the predicted interaction probabilities
and a one-hot encoding of the current energy bin, making predictions
interaction-specific rather than globally regressed.

**N-step bootstrapping.** Beam Weaver uses a custom n-step replay
buffer. The critic target employs an n-step return rather than a purely
one-step backup:

$$y_t^{(n)} = r_t^{(n)} + \gamma^n \bigl( Q_{\mathrm{target}}(s_{t+n}, a_{t+n}) - \alpha \log \pi(a_{t+n} \mid s_{t+n}) \bigr),$$

where $r_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k}$ is the
accumulated n-step reward. This choice was motivated by the sequential
nature of photon histories: a photon history is a chain of physically
coupled interaction decisions (such as successive Compton scatterings in
a shower), so multi-step targets help the critic connect local
decisions to downstream transport consequences more effectively than a
purely one-step backup, while remaining more stable than full Monte
Carlo returns.

**Teacher-forced replay storage.** During teacher-forced curriculum
phases (described below), the replay buffer stores the overridden
MC-consistent action rather than the raw action proposed by the agent.
This is a deliberate design choice: the off-policy dataset seen by the
critic and the auxiliary physics losses is anchored to the MC reference
during curriculum stages, and only later becomes fully policy-driven as
teacher forcing is withdrawn.

# Training workflow

Beam Weaver is trained in two main stages: (I) supervised pretraining
of the physics branch on MC-generated labels, and (II) curriculum-based
off-policy SAC with staged teacher forcing. This section describes both
stages in detail.

## Stage I: physics-head pretraining

Before SAC optimization begins, Beam Weaver generates a dataset of
event-level transport targets by running the internal MC simulator. For
each sampled interaction, the code stores an observation vector
together with the following physical targets: free path, interaction
class, photon scattering angle, outgoing photon energy, number of
secondaries, and secondary-particle energies and angles.

The observation vector is z-scored before pretraining. Physical targets
are transformed into numerically convenient representations:
interaction class and number of secondaries are kept as discrete
labels; free path and energies are mapped as $\log(1+x)$; angular
quantities are represented as $(\sin\theta, \cos\theta)$ pairs.

The physics branch is a structured multi-head module attached to the
shared feature extractor. In the current implementation, the shared
feature extractor maps the observation to a latent vector; a physics
backbone processes that latent representation; and four prediction heads
produce outputs: a process head (interaction class), an energy head
(interaction-conditional energy quantities), an angle head
(interaction-conditional angular quantities), and a secondary-count
head (number of secondaries). The energy and angle heads are
conditioned on both the latent physics representation and additional
context: the predicted interaction probabilities and a one-hot encoding
of the current energy bin. This means the physics branch is trained as
a conditional predictor of interaction-specific transport kernels
rather than as a single global regressor.

The pretraining loss is a weighted sum of four terms:

$$\mathcal{L}_{\mathrm{pre}} = w_P \,\mathcal{L}_P + w_E \,\mathcal{L}_E + w_\theta \,\mathcal{L}_\theta + w_N \,\mathcal{L}_N,$$

with equal default weights $w_P = w_E = w_\theta = w_N = 0.25$. Here
$\mathcal{L}_P$ is a cross-entropy loss for the interaction type;
$\mathcal{L}_E$ is a masked mean-squared-error loss over
$\log(1 + \mathrm{fp})$, $\log(1 + E_{\mathrm{out}})$, and the
secondary log-energies; $\mathcal{L}_\theta$ is a masked
mean-squared-error loss over the sine/cosine representation of photon
and secondary angles; and $\mathcal{L}_N$ is a cross-entropy loss for
the number of secondaries. Secondary-particle losses are masked so that
non-existent secondaries (for example, in Rayleigh or photoelectric
events that produce no photon-channel secondaries) do not contribute to
the objective. A small Gaussian perturbation is added to normalized
observations during pretraining for robustness. The optimizer is Adam
with weight decay [@kingma2015adam], and validation loss is monitored
with a ReduceLROnPlateau scheduler.

The pretraining stage produces a checkpoint
(`physics_head_pretrained.pth`) that initializes the physics branch
before SAC training begins. This pretrained branch is then loaded as an
informed initialization and initially frozen to stabilize the
subsequent reinforcement-learning stages.

## Stage II: curriculum-based SAC

After pretraining, Beam Weaver enters a curriculum-based hybrid SAC
training stage [@bengio2009curriculum]. The curriculum is organized into
five sequential phases, each with a specific learning objective,
trainable subset, and teacher-forcing schedule. Training does not
optimize all policy components simultaneously; instead, different heads
are frozen or activated according to the current phase.

### Phase 0: discrete interaction learning, fully teacher-forced

Only the shared feature extractor and the discrete head are trainable.
The discrete interaction choice is overridden at every step by the MC
reference (full teacher forcing). At the start of this phase, the actor
initializes the discrete logits from physics-derived interaction
probabilities and the mean-free-path component from the tabulated
physics reference; the mean-free-path residual output and its
corresponding variance are then frozen. The reward exposed to SAC
during this phase is only the discrete interaction reward
$r_{\mathrm{disc},t}$ (described below).

### Phase 1: discrete interaction learning, teacher forcing decays

The same trainable subset is maintained, but discrete teacher forcing
is linearly decayed over a fixed horizon. This phase implements
scheduled teacher forcing: the agent transitions from MC-labelled
interaction selection toward autonomous discrete policy learning. The
reward remains $r_{\mathrm{disc},t}$.

### Phase 2: continuous transport learning, fully teacher-forced

The discrete head is frozen. The continuous transport heads (energy,
angle, secondary-particle outputs) become trainable, while the
mean-free-path output remains fixed to its physics initialization.
During this phase, the MC kernel values are forced into the stored
action representation, so the replay buffer contains MC-consistent
targets for continuous transport quantities. The reward shifts to
$r_{\mathrm{kernel},t} + r_{E\text{-corr},t}$ (described below).

### Phase 3: continuous transport learning, teacher forcing decays

The same branch structure as Phase 2 is maintained, but kernel-level
teacher forcing is gradually reduced. The agent transitions from
MC-driven kernel supervision to autonomous continuous prediction.

### Phase 4 and beyond: hybrid refinement

The discrete head and mean-free-path output both remain frozen. The
remaining continuous transport heads (energy loss, angles,
secondary-particle energy) continue to be optimized under the kernel
reward structure.

## Reward design

Beam Weaver does not use a single stationary reward throughout
training. Instead, the reward is phase-dependent and switched
explicitly by the curriculum wrapper.

### Discrete interaction reward (Phases 0--1)

During the discrete curriculum, the goal is to reproduce the correct
interaction mixture $\pi_{\mathrm{true}}(a \mid E)$ on the energy
grid. The code maintains an exponentially weighted moving-average
(EWMA) histogram $H_{b,a}$ of recent agent-selected discrete actions
per energy bin $b$, with decay factor $\lambda_{\mathrm{hist}} =
0.995$, together with a cumulative histogram $C_{b,a}$ for diagnostics
and bin-validity checks (only bins with at least 10 cumulative events
are used). The weighted target mixture is

$$p_{\mathrm{true},a} = \frac{\sum_b w_b\, \pi_{\mathrm{true},b,a}}{\sum_b w_b},$$

where $w_b = \mathbf{1}[\sum_a C_{b,a} \ge 10] \sum_a H_{b,a}$ and
$\pi_{\mathrm{true},b,a}$ is the tabulated interaction probability in
energy bin $b$. The discrete reward for the chosen action $a_t$ is a
log-likelihood ratio against a uniform baseline:

$$r_{\mathrm{disc},t} = \log\!\left(\frac{p_{\mathrm{true},a_t} + \varepsilon}{1/4 + \varepsilon}\right),$$

with small numerical $\varepsilon$. This favours interaction choices
that are more probable under the physics mixture than under a uniform
prior. For monitoring only (not added to the reward), the code also
logs the Jensen--Shannon divergence and the $L^1$ distance between the
agent histogram and the target mixture.

### Kernel reward (Phases 2+)

Once the discrete curriculum is complete, the reward shifts to a
continuous/kernel objective. The full kernel reward at each step is

$$r_{\mathrm{kernel},t} = r_{\mathrm{dist},t} + r_{\phi,t} + r_{\phi_e,t} + r_{\theta,\mathrm{pair},t} + r_{E,\mathrm{pair},t} + r_{e,\mathrm{comp},t},$$

clipped to $[-50, 50]$, where each term targets a specific physical
constraint. The components are as follows.

**Distribution-matching term.** For each interaction type $i$, the code
evaluates a process-specific acceptance kernel $A_i(\cdot)$ through
`accept_prob(...)`. The primary angular reward is

$$r_{\mathrm{dist},t}^{(i)} = \log\!\left(\frac{A_i + \varepsilon}{1/180 + \varepsilon}\right),$$

where the baseline $1/180$ reflects a uniform angular histogram
discretization. This term rewards predicted angles that land in
high-density regions of the physical scattering kernel.

**Per-bin KL angular-distribution penalty.** For each energy bin and
interaction type, Beam Weaver maintains a short history of generated
angles and compares the empirical histogram against the target angular
distribution. When at least 20 angles have been accumulated in a given
energy--interaction bin, the code computes the KL divergence

$$D_{\mathrm{KL}}(p_{\mathrm{target}} \| p_{\mathrm{empirical}}) = \sum_j p_{\mathrm{target},j} \log\!\left(\frac{p_{\mathrm{target},j}}{p_{\mathrm{empirical},j}}\right),$$

and adds the penalty $r_{\mathrm{KL},t}^{(i)} = -0.05\,
\mathrm{clip}(D_{\mathrm{KL}},\, 0,\, 10)$ to the distribution term.

**Rayleigh energy consistency.** For Rayleigh scattering (elastic), the
outgoing photon energy must equal the incoming energy. A penalty
proportional to the relative energy error is applied:
$r_{E,\mathrm{Rayleigh},t} = -0.01\, |E_\gamma^{\mathrm{pred}} -
E_\gamma^{\mathrm{in}}| / \max(E_\gamma^{\mathrm{in}}, 10^{-6})$.

**Compton electron kinematic consistency.** For Compton scattering, the
code derives the reference recoil-electron kinematics from energy--momentum
conservation applied to the predicted photon branch:
$E_{e,\mathrm{MC}} = E_\gamma^{\mathrm{in}} -
E_\gamma^{\mathrm{pred}}$, with the electron momentum and polar angle
obtained from relativistic kinematics. The consistency penalty is

$$r_{e,\mathrm{comp},t} = -\left[0.05\,\frac{|E_e^{\mathrm{pred}} - E_{e,\mathrm{MC}}|}{\max(E_{e,\mathrm{MC}}, 10^{-6})} + 0.05\,\frac{|\theta_e^{\mathrm{pred}} - \theta_{e,\mathrm{MC}}|}{\pi}\right].$$

**Pair-production energy-sharing and angular consistency.** For pair
production, the code combines angular-distribution matching, an
energy-sharing penalty $r_{E,\mathrm{pair},t} = -0.05\,
|E_e^{\mathrm{pred}} + E_p^{\mathrm{pred}} -
(E_\gamma^{\mathrm{in}} - 2m_e c^2)| / (E_\gamma^{\mathrm{in}} -
2m_e c^2)$, and a small-angle consistency term comparing the predicted
electron polar angle against a reference model.

**Azimuthal uniformity.** For processes where the azimuthal angle
should be uniformly distributed, Beam Weaver uses the first circular
moment $R = |N^{-1} \sum_{n=1}^{N} e^{i\phi_n}|$ and applies the
penalty $r_\phi = -0.05\, R$. This is evaluated over batches of 32
accumulated samples and applied to photon azimuth (Rayleigh, Compton)
and electron azimuth (Compton, photoelectric, pair).

**Forbidden photon-energy penalty.** For photoelectric absorption and
pair production, the outgoing photon energy should be zero. The code
adds a hard penalty $r_{E\text{-corr},t} = -5$ whenever
$E_\gamma^{\mathrm{pred}} > 10^{-12}$ for these interaction types.

### Reward exposure by phase

The curriculum wrapper controls which reward components the agent sees:

- **Phases 0--1:** $R_t = r_{\mathrm{disc},t}$
- **Phases 2+:** $R_t = r_{\mathrm{kernel},t} + r_{E\text{-corr},t}$

The curriculum therefore does not merely freeze and unfreeze different
heads; it also changes which part of the objective the agent actually
receives.

## Actor loss during RL

During reinforcement learning, Beam Weaver does not optimize a pure SAC
objective. The actor loss is multitask:

$$\mathcal{L}_{\mathrm{actor}} = \mathcal{L}_{\mathrm{SAC}} + \lambda_{\mathrm{phys}} \,\mathcal{L}_{\mathrm{phys}},$$

where
$\mathcal{L}_{\mathrm{SAC}} = \mathbb{E}[\alpha \log \pi(a \mid s) - Q(s, a)]$
is the standard SAC actor loss and $\mathcal{L}_{\mathrm{phys}}$ is an
auxiliary physics-regularization term composed of interaction-aware
energy loss, angular loss, sine/cosine unit-norm regularization (weight
0.4), number-of-secondaries classification loss, and interaction-type
classification loss. The physics branch therefore continues to provide
supervised guidance throughout RL training, making Beam Weaver a
physics-regularized hybrid SAC framework rather than a purely
reward-driven SAC implementation.

## Entropy control

Beam Weaver retains SAC's automatic entropy-coefficient tuning and
tracks discrete and continuous entropy separately throughout training.
This matters because exploration requirements differ between the
discrete curriculum stages (where the agent must learn to select among
four interaction types across the energy spectrum) and the later
kernel-learning stages (where it must match continuous angular and
energy distributions).

# Evaluation and diagnostics

The public release includes evaluation utilities that compare MC
reference behaviour and agent behaviour through several complementary
diagnostics:

- **Interaction-frequency histograms:** energy-binned comparison of the
  discrete interaction mixture between the agent and the MC reference.
- **Angular distributions:** per-interaction, per-energy-bin histograms
  of predicted scattering angles compared against the target
  process-specific kernels.
- **Secondary-particle summaries:** statistics on secondary-electron
  energies, multiplicities, and angular distributions.
- **Particle tracks:** visual inspection of individual photon histories
  generated by the agent and by MC, enabling qualitative comparison of
  shower structure.
- **Percentage depth-dose curves:** comparison of depth-dose profiles
  between the agent and the MC reference for a specified beam energy and
  geometry, using either condensed-history electron transport or local
  energy deposition.

# Current limitations

This public release should be understood as a research snapshot with
the following known limitations:

- The codebase is currently monolithic (a single Python script) rather
  than modular.
- Execution is interactive rather than command-line driven.
- File paths are largely hard-coded and root-relative.
- The geometry and material scope are intentionally narrow (homogeneous
  water phantom only).
- Angular distribution fine-tuning for the continuous transport heads
  remains the principal unfinished component.
- Validation remains at the proof-of-concept level rather than
  production-grade.
- Redistribution of some external data tables may depend on their
  original licence terms.

# Research impact statement

Beam Weaver's immediate impact is as an openly archived, reproducible
baseline for research on learned photon transport. The software makes a
complete experimental workflow publicly inspectable: generation of
MC-derived targets, physics-head pretraining, curriculum-based
reinforcement-learning training, checkpointing, and evaluation against
reference transport behaviour. This lowers the barrier for other
researchers who want to benchmark alternative policies, reward
functions, action parameterisations, or surrogate-model designs without
rebuilding the full transport-and-training pipeline from scratch.

The software also has credible near-term significance because it
focuses on a research question that is not well served by either
production Monte Carlo codes or generic reinforcement-learning
libraries alone: can a policy learn the stochastic event structure of
radiation transport, not merely its final aggregate outputs? Beam
Weaver provides a concrete answer path by exposing both agent and
reference transport at the level of interaction class, secondary
generation, trajectory evolution, and depth-dose observables in water.

At the current project stage, the software already supports physically
meaningful interaction simulation and comparison workflows in the
target water-phantom setting. Even in its current state---functional
but with angular fine-tuning still in progress---the release is useful
as a research object because it captures a distinctive architecture, an
unusual training regime, and a transport-specific evaluation strategy
that would otherwise remain unavailable. Releasing the code now is a
deliberate invitation to visibility, reuse, criticism, and eventual
development collaboration, rather than a claim that the project is
complete. The Zenodo archive and public repository make the software
citable, inspectable, and reusable in methodological work
[@beamweaver_software].

# AI usage disclosure

Generative AI tools were used to assist with drafting parts of the
repository documentation and an initial draft of this manuscript.
Specifically, OpenAI ChatGPT (GPT-4o and GPT-5.4 Thinking) was used
for language editing and draft generation, and Anthropic Claude
(Sonnet 4.5 and Opus 4.6) was used for code review, technical editing,
and solution development. The human authors reviewed, corrected, and
validated all text, software descriptions, and scholarly claims, and
they remain fully responsible for the final submission. All code was
validated by the human authors.

# Acknowledgements

Beam Weaver builds on the broader open scientific software ecosystem,
including PENELOPE for photon and electron transport methodology
[@penelope2018], Geant4 for transport-system design inspiration
[@agostinelli2003geant4], Stable-Baselines3 for reinforcement-learning
infrastructure [@raffin2021stable], and PyTorch for the underlying
deep-learning framework [@paszke2019pytorch]. Beam Weaver also draws
on the reinforcement-learning literature on multi-step bootstrapping
and entropy-regularized policy optimization [@suttonbarto2018;
@haarnoja2018sac], as well as Hubbell and co-workers for atomic form
factors and incoherent scattering functions [@hubbell1975].

# References
