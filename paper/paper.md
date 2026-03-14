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
event-by-event photon transport in a homogeneous water phantom with a
physics-informed hybrid Soft Actor-Critic (SAC) agent. The software
combines three tightly coupled components: an internal Monte Carlo (MC)
reference simulator, a reinforcement-learning environment that exposes
transport as a structured sequential decision problem, and a custom
multi-head SAC implementation that predicts both a discrete interaction
type and continuous transport quantities such as free path, scattering
geometry, and secondary-particle properties.

In contrast to standard SAC implementations [@haarnoja2018sac], Beam
Weaver uses an n-step replay and training workflow [@suttonbarto2018] to
propagate information across short transport sequences that are
physically meaningful for photon histories. The current release models
the four dominant photon interaction cross sections ---
Rayleigh scattering, Compton scattering, photoelectric
absorption, and pair production---together with optional simplified
condensed-history secondary-electron transport for depth-dose
estimation. Alternatively it deposits the energy locally without electron transport.

Beam Weaver is released as a transparent research platform for studying
whether reinforcement learning can reproduce the stochastic event
structure of radiation transport while retaining physically interpretable
behavior. The software is already functional in its target
water-phantom setting, although it remains a work in progress: core
interaction simulation is operational, while angular fine tuning remains
the main unfinished component. Releasing Beam Weaver in its current form
makes its unusual architecture, training workflow, and evaluation
pipeline available to the community for scrutiny, reuse, and eventual
collaboration. The public software archive for the release described
here is available at [@beamweaver_software].

# Statement of need

High-fidelity radiation transport is typically performed with mature
Monte Carlo codes such as PENELOPE, Geant4, EGSnrc, and related
toolchains [@penelope2018; @agostinelli2003geant4; @kawrakow2000egsnrc].
These remain the standard for validated transport and dosimetric
accuracy, but their computational cost can limit rapid experimentation
and repeated evaluation in algorithm-development settings. At the same
time, reinforcement-learning libraries such as Stable-Baselines3 provide
robust algorithmic infrastructure, but they do not include
radiation-physics environments, transport samplers, or validation
workflows tailored to stochastic photon transport [@raffin2021stable].

Beam Weaver was developed to address this gap. It is intended for
researchers in medical physics, computational radiation transport, and
scientific machine learning who want an open platform for investigating
learned transport surrogates at the event level. Rather than learning a
final dose map directly, Beam Weaver models the transport process itself:
the agent observes photon state variables and cross-section-derived
quantities, then predicts the next interaction class together with
continuous outcomes that can be compared against MC-generated reference
behavior.

The release described here is also motivated by reproducibility and
community visibility. Beam Weaver bundles the workflow needed to
generate MC-derived supervision targets, pretrain a physics branch,
train the hybrid SAC policy, save checkpoints, and compare learned
transport behavior against MC references through interaction
statistics, secondaries, tracks, and depth-dose curves. It is therefore
not merely a model checkpoint, but an inspectable research pipeline for
learned transport. Making that pipeline public at this stage is valuable
even though the project remains unfinished, because the software already
demonstrates a distinctive combination of structured RL actions,
physics-guided supervision, curriculum training, and n-step SAC updates.

# State of the field

Established transport packages such as PENELOPE, Geant4, and EGSnrc provide
validated and extensible particle-transport capabilities across broad classes of
materials, geometries, and interaction models [@penelope2018;
@agostinelli2003geant4; @kawrakow2000egsnrc]. They are the appropriate tools
when the goal is production simulation or validated reference calculation. Beam
Weaver does not compete with them on scope, maturity, or clinical readiness.
Instead, it uses a deliberately narrower setting---$10 \times 10$ cm$^2$ photon beam transport in a
$100 \times 100 \times 100$ cm$^3$---to study whether a SAC agent can learn
the event-level structure of radiation transport from a physics-based simulator.

General-purpose reinforcement-learning frameworks likewise address a different
problem. Stable-Baselines3 and related toolkits provide training algorithms,
policies, buffers, and utilities, but they leave domain modelling to the user
[@raffin2021stable]. They do not include photon interaction models, particle
kinematics, dose tallies, or transport-specific diagnostics. Standard SAC
formulations are also not designed around the temporally structured reward and
credit-assignment patterns that arise in transport histories
[@haarnoja2018sac]. Beam Weaver builds on that ecosystem by contributing a
transport-specific environment, a hybrid action-space policy, a physics-guided
pretraining stage, and an n-step replay formulation designed for 
interaction sequences in stochastic transport.

Beam Weaver was therefore built rather than implemented as a contribution to an
existing transport code or a generic reinforcement-learning package for three
reasons. First, the design goal is methodological experimentation with learned
transport, not an extension of a production Monte Carlo code. Second, the software
needs tight coupling between MC-generated labels, staged reinforcement-learning
curricula, and transport-specific reward shaping. Third, the code exposes
intermediate physical quantities---interaction statistics, secondaries, tracks,
and depth-dose curves---that are especially useful for comparing learned and
reference transport process by process.

# Software design

Beam Weaver is organized around three tightly coupled components. The
first is a compact MC reference simulator written in Python. The
simulator uses cross-section tables derived from the PENELOPE
material-data files [@penelope2018] and samples the four photon
interactions which are used by the reinforcement-learning environment:
Rayleigh scattering, Compton scattering, photoelectric absorption, and
pair production. For secondary charged-particle handling, the present
implementation also draws methodological inspiration from both
PENELOPE and Geant4 condensed-history ideas
[@penelope2018; @agostinelli2003geant4], but provides the user with a simpler local energy deposition approach as well. 

The second component is a reinforcement-learning environment that casts
transport as a structured sequential decision problem. Observations
encode photon energy, position, direction, and cross-section-derived
quantities. Actions are hybrid: the agent selects one of the discrete
interaction classes and predicts continuous transport outputs such as
free path, scattering geometry, and secondary-particle parameters. The
environment also exposes transport-specific diagnostics and rewards,
including energy-binned interaction statistics, angular-distribution
comparisons, secondary summaries, and percentage-depth-dose behavior.
These diagnostics are important because Beam Weaver is intended not only
to optimize a scalar score, but to compare learned and reference
transport process by process.

The third component is a custom SAC training stack built on
Stable-Baselines3 [@raffin2021stable]. Beam Weaver extends the standard
SAC workflow with a hybrid actor-critic architecture, n-step replay with bootstrapping, a
physics-pretraining stage, and curriculum-style phase scheduling. The
policy is multi-head: a shared encoder feeds (i) a discrete SAC head for
interaction selection, (ii) continuous heads for interaction-conditional
transport variables, and (iii) a physics branch composed of auxiliary
heads for quantities such as free path, outgoing energy, angular
variables, interaction type, and secondary multiplicity. This design
makes the implementation materially different from off-the-shelf SAC
[@haarnoja2018sac].

Training proceeds in stages. First, the code generates MC-derived
supervision targets and pretrains the physics branch using supervised
losses on physically meaningful quantities. In the current public
workflow, this pretrained branch is then loaded as an informed
initialization and initially frozen to stabilize the subsequent
reinforcement-learning stages. The early SAC curriculum focuses on the
discrete interaction head, which is trained to reproduce the correct
interaction mixture through energy-binned histogram similarity against
reference MC behavior. During this stage, teacher-forced supervision is
used and then progressively worn off, providing a curriculum from
strongly supervised interaction selection toward autonomous policy
learning.

After the discrete interaction policy has been shaped, the training
schedule shifts toward the continuous transport heads. These heads are
trained in an analogous staged manner, again with teacher-forced
guidance that is gradually removed, but now targeting free path, energy
transfer, angular behavior, and secondary-particle outputs. The actor
objective remains physics-regularized during RL optimization through
auxiliary supervised losses, so Beam Weaver is better understood as a
physics-informed hybrid SAC framework than as a purely reward-driven SAC
baseline.

A further distinction from standard SAC is explicit entropy management
across curriculum phases together with the use of n-step bootstrapped
targets [@suttonbarto2018]. The entropy coefficient is controlled so that
exploration remains appropriate as the task shifts between discrete and
continuous transport learning, while the n-step return improves reward
propagation across short but physically meaningful transport histories.
This combination is particularly useful in stochastic transport, where a
local interaction choice can influence several downstream event
consequences rather than only an immediate scalar reward.

The public release also includes utilities for checkpointing,
interaction-frequency analysis, secondary-particle summaries,
trajectory inspection, and percentage-depth-dose comparison against MC
references. Beam Weaver is therefore not just a trained policy, but a
complete experimental workflow for studying learned transport in a
transparent and reproducible way.

Beam weaver uses the PENELOPE cross-sections for Photoelectric and Compton and pair-production (stored in `Final_cross_sections.csv`). The energy binning of these tables becomes the in-built energy grid for training and learning.
Rayleigh scattering is also taken from the PENELOPE tables (stored in `Rayleigh_cross_sections.csv`), the binning is different, reflecting the way the table appears in PENELOPE. The binning is however mapped to the binning of the other cross-sections'file.

For the incoherent scattering function $S(q)$, Beam weaver uses
Hubbell-style Waller--Hartree tabulations for hydrogen and oxygen
[@hubbell1975]. The water table used by Beam Weaver is constructed through the
independent-atom stoichiometric sum

$S_{\mathrm{H_2O}}(x)=2S_{\mathrm H}(x)+S_{\mathrm O}(x)$,

where 

$x=\sin(\theta/2)/\lambda$. 

For compatibility with the current code,
this quantity is stored in `water_sq.csv` under the legacy column name `q`,
while the sampler internally converts the full momentum transfer to the
Hubbell table variable before interpolation.

For the coherent form factor $F(q)$, Beam Weaver uses the IAEA compact three-term
Gaussian approximation for water,

$F(q)=\sum_{i=1}^{3} a_i \exp\!\left[-b_i\left(\frac{q}{4\pi}\right)^2\right],$

with $a=\{0.4899,\,0.2626,\,0.2254\}$ and
$b=\{1.4752,\,4.1567,\,15.8047\}$. Rayleigh angles are then accepted with a
weight proportional to


$\frac{F(q)^2}{F(0)^2}\,\frac{1+\cos^2\theta}{2},$

i.e. a Thomson-like angular factor modulated by the coherent form factor.
In the current public release, these coefficients are implemented directly
in the code as a compact water-specific fit rather than loaded from a
tabulated molecular form-factor database.



# Research impact statement

Beam Weaver's immediate impact is as an openly archived, reproducible baseline
for research on learned photon transport. The software makes a complete
experimental workflow publicly inspectable: generation of MC-derived targets,
physics-head pretraining, reinforcement-learning training, checkpointing, and
evaluation against reference transport behaviour. This lowers the barrier for
other researchers who want to benchmark alternative policies, reward functions,
action parametrisations, or surrogate-model designs without rebuilding the full
transport-and-training pipeline from scratch.

The software also has credible near-term significance because it focuses on a
research question that is not well served by either production Monte Carlo codes
or generic reinforcement-learning libraries alone: can a policy learn the
stochastic event structure of radiation transport, not merely its final
aggregate outputs? Beam Weaver provides a concrete answer path by exposing both
agent and reference transport at the level of interaction class, secondary
generation, trajectory evolution, and depth-dose observables in water.

At the current project stage, the software already supports physically
meaningful interaction simulation and comparison workflows in the target
water-phantom setting. The main remaining refinement is angular fine tuning.
Even in this paused but functional state, the release is useful as a research
object because it captures a distinctive architecture, an unusual training
regime based on n-step SAC, and a transport-specific evaluation strategy that
would otherwise remain unavailable to the community. Releasing the code now is
therefore a deliberate invitation to visibility, reuse, criticism, and eventual
development collaboration, rather than a claim that the project is finished. The
Zenodo archive and public repository further make the software citable,
inspectable, and reusable in methodological work [@beamweaver_software].



# AI usage disclosure

Generative AI was used to assist with drafting parts of the repository
documentation and an initial draft of this JOSS manuscript. Specifically,
OpenAI's ChatGPT model GPT-5.4 Thinking was used for language editing and draft
generation. The human authors reviewed, corrected, and validated all text,
software descriptions, and scholarly claims, and they remain fully responsible
for the final submission.

Generative AI was used to help build the python code. 
We used ChatGPT model GPT-4o, and Claude Sonnet 4.5 to review code and propose solutions. 
All code was humanly validated.

# Acknowledgements

Beam Weaver builds on the broader open scientific software ecosystem, including
PENELOPE for photon and electron transport methodology [@penelope2018], Geant4
for transport-system design inspiration [@agostinelli2003geant4], and
Stable-Baselines3 for reinforcement-learning infrastructure [@raffin2021stable].
Beam Weaver also draws on the reinforcement-learning literature on
multi-step bootstrapping and entropy-regularized policy optimization
[@suttonbarto2018; @haarnoja2018sac].

# References
