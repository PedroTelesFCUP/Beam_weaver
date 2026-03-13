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

Beam Weaver is an open-source Python research framework for learning event-by-event
photon transport in a homogeneous water phantom with a modified n-step Soft-Actor-Critic agent.
The software combines a custom Monte Carlo (MC) reference simulator, a
physics-informed environment for reinforcement learning, and a hybrid
Soft Actor-Critic (SAC) policy that predicts both a discrete interaction class
and continuous transport quantities such as free path, scattering geometry, and
secondary-particle properties. In contrast to standard SAC implementations
[@haarnoja2018sac], Beam Weaver uses an n-step training workflow
[@suttonbarto2018] to propagate transport information across short event
sequences that are physically meaningful for photon histories. The current
release focuses on the four dominant photon interactions implemented in the
codebase---Rayleigh scattering, Compton scattering, photoelectric absorption,
and pair production---together with simplified secondary-electron transport for
depth-dose estimation.

Beam Weaver is a proof-of-concept, a transparent experimental platform for studying
whether modern reinforcement-learning methods can emulate stochastic
radiation transport while preserving physically interpretable behaviour. The
software is released in a substantially working but still evolving state: core
interaction simulation is already operational in the target water-phantom
setting, while angular fine tuning remains the main unfinished component.
Releasing Beam Weaver now makes its unusual problem setting, data flow, and
evaluation workflow available to the community, with the explicit goal of
supporting scrutiny, reuse, and future collaboration. The public software
archive for the release described here is available at [@beamweaver_software].

# Statement of need

High-fidelity radiation transport is usually performed with mature Monte Carlo
codes such as PENELOPE, Geant4, EGSnrc, and MCNP-class toolchains
[@penelope2018; @agostinelli2003geant4; @kawrakow2000egsnrc]. These codes are
the standard for accuracy and remain indispensable for dosimetry, algorithm
benchmarking, and method development, but their computational cost can limit
rapid experimentation and make repeated transport evaluations expensive. At the
same time, reinforcement-learning libraries such as Stable-Baselines3 provide
robust implementations of algorithms like SAC, but they do not provide
radiation-physics environments, transport samplers, or validation workflows
tailored to stochastic particle transport [@raffin2021stable].

Beam Weaver was developed to address this gap. Its target audience is
researchers in medical physics leaning strongly towards computation, computational radiation transport, and
scientific machine learning who want an open platform for testing learned
transport surrogates at the event level. Instead of learning a final dose map
directly, Beam Weaver models the transport process itself, taking advantage of t
he fact that the Monte Carlo simulation itself produces billions of events that can 
be used as reinforcement learning observables, and function as ground-truth in supervised learning ecosystems. 
The agent observes photon state variables and material-dependent quantities, 
then predicts the next interaction class together with continuous outcomes that can 
be compared against MC-generated references. 

The release described here is also motivated by reproducibility and community
visibility. The repository bundles the workflow needed to generate MC-derived
supervision targets, pretrain a physics head, train the hybrid SAC policy, save
checkpoints, and compare learned trajectories and depth-dose behaviour against
MC references. Beam Weaver is therefore not just a model checkpoint; it is an
inspectable research pipeline for learned transport. Making that pipeline public
at this stage is valuable even though the project remains a work in progress,
because the software already demonstrates a distinctive and non-standard
combination of radiation transport, structured RL actions, physics-guided
pretraining, and n-step SAC training.

# State of the field

Established transport packages such as PENELOPE, Geant4, and EGSnrc provide
validated and extensible particle-transport capabilities across broad classes of
materials, geometries, and interaction models [@penelope2018;
@agostinelli2003geant4; @kawrakow2000egsnrc]. They are the appropriate tools
when the goal is production simulation or validated reference calculation. Beam
Weaver does not compete with them on scope, maturity, or clinical readiness.
Instead, it uses a deliberately narrower setting---monoenergetic 10x10 cm^2 photon beam
transport in a 100x100x100 cm^3 box of water---to study whether a SAC agent can learn
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

Beam Weaver is organized around three main components. The first is a
PENELOPE-derived MC reference simulator written in Python. The simulator uses
cross-section tables derived from the PENELOPE material-data workflow
[@penelope2018] and samples the same four photon interaction channels used by
the reinforcement-learning environment, using heavily inspired algorithms from PENELOPE
and GEANT4 (if the user choses to explicitly simulate the secondary particles with a condensed history model). 
[@penelope2018; @agostinelli2003geant4]. Beam Weaver does not claim to reproduce
the full scope or validation status of either code system; rather, it adapts
selected ideas from those toolkits into a compact research prototype that makes
the learned-transport problem tractable and inspectable.

The second component is the environment that wraps the transport process as a
reinforcement-learning problem. Observations encode photon energy, position,
direction, and cross-section-derived quantities. Actions are hybrid: the policy
selects one of the discrete interaction types and predicts continuous transport
outputs such as free path, scattering angles, or secondary-particle parameters.
The environment is paired with curriculum-style phase scheduling so that early
training focuses on interaction-type learning before later phases emphasise
continuous transport outputs, consistency constraints, and richer trajectory
statistics.

The third component is a custom SAC implementation and associated training
workflow. Beam Weaver extends Stable-Baselines3 with an n-step replay buffer, a
hybrid actor-critic policy architecture, and a pretrained physics head. This
makes the learning procedure materially different from off-the-shelf SAC
[@haarnoja2018sac]. The n-step return is particularly useful in Beam Weaver
because transport decisions often have immediate local consequences but also
affect short downstream event chains, for example through Compton showers finishing typically with a photoelectric interaction. 
The structured action space and physics head likewise narrow the search space while still allowing the policy to
learn stochastic transport behaviour from experience rather than from hard-coded
rules alone.

The public release also includes utilities for interaction frequencies,
secondary-particle summaries, trajectory inspection, and percentage-depth-dose
(PDD) comparison. These tools matter because Beam Weaver is intended to support
methodological studies of learned transport, where diagnostics beyond a single
scalar score are essential for judging physical plausibility.

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

# Acknowledgements

Beam Weaver builds on the broader open scientific software ecosystem, including
PENELOPE for photon and electron transport methodology [@penelope2018], Geant4
for transport-system design inspiration [@agostinelli2003geant4], and
Stable-Baselines3 for reinforcement-learning infrastructure [@raffin2021stable].

# References
