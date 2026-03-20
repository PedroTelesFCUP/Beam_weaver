---
title: 'Beam Weaver -- a physics-informed n-step Soft Actor-Critic agent for event-level photon transport in water'
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

Beam Weaver is an open-source Python framework for learning event-by-event photon transport in a homogeneous water phantom using a physics-informed hybrid Soft Actor-Critic (SAC) agent. It combines three tightly coupled components: (i) a Monte Carlo (MC) reference simulator using PENELOPE-derived cross-section tables [@penelope2018], (ii) a Gymnasium-compatible reinforcement-learning (RL) environment that casts photon transport as a sequential decision problem, and (iii) a custom multi-head SAC that jointly predicts a discrete interaction type and continuous transport quantities---free path, scattering geometry, outgoing energy, and secondary-particle properties. Unlike standard SAC [@haarnoja2018sac], Beam Weaver uses $n$-step bootstrapped returns [@suttonbarto2018], a five-phase curriculum with scheduled teacher forcing, and physics-informed pretraining. \autoref{fig:architecture} shows the full actor-critic architecture.

![Beam Weaver actor-critic architecture. The observation vector feeds a shared residual MLP that branches into a discrete head (interaction type), continuous heads (angles, energies, secondaries), and a pretrained physics branch. The mean free path is hard-coded from tabulated cross sections. A curriculum wrapper selects the phase-appropriate reward, and teacher forcing overwrites replay-buffer actions during early phases.\label{fig:architecture}](fig_architecture.pdf)

# Statement of need

High-fidelity radiation transport relies on mature MC codes such as PENELOPE, Geant4, and EGSnrc [@penelope2018; @agostinelli2003geant4; @kawrakow2000egsnrc], while RL libraries like Stable-Baselines3 [@raffin2021stable] provide algorithmic infrastructure but no radiation-physics environments. Existing ML approaches to accelerating radiation transport focus on surrogate models for aggregate observables---dose maps, depth-dose curves, or MC denoising---rather than on learning the event-level transport process itself.

Beam Weaver addresses a different question: can an RL agent learn to reproduce the stochastic sequence of interaction decisions that constitutes a photon history? The agent observes photon state variables and cross-section data, then predicts the next interaction class together with continuous outcomes that can be compared against MC references process by process.

The release bundles the complete workflow: MC data generation, physics-head pretraining, curriculum-based SAC training, checkpointing, and evaluation through interaction statistics, secondary-particle summaries, tracks, and percentage depth-dose curves---making it an inspectable research pipeline, not merely a model checkpoint.

# State of the field

Established MC codes provide validated photon transport across broad material and geometry classes [@penelope2018; @agostinelli2003geant4; @kawrakow2000egsnrc]. Beam Weaver does not compete on scope or clinical readiness; it uses a deliberately narrow setting---a $10 \times 10$~cm$^2$ beam on a $100 \times 100 \times 100$~cm$^3$ water phantom---to study whether a SAC agent can learn event-level transport from a physics-based simulator.

SAC was chosen because its maximum-entropy formulation produces inherently stochastic policies (physically appropriate when multiple interactions carry non-zero probability), it is off-policy (essential for learning from teacher-forced replay buffers), it handles continuous action spaces natively, and its automatic entropy tuning adapts exploration across curriculum phases with different natural entropy scales.

# Software design

Beam Weaver is organised around three components, currently distributed as a single Python research script with accompanying physics tables and metadata.

**MC reference simulator.** A compact Python simulator samples Rayleigh, Compton, photoelectric, and pair-production interactions using PENELOPE-derived cross sections [@penelope2018], Hubbell incoherent scattering functions and coherent form factors [@hubbell1975], and EPDL photoelectric shell data [@cullen1997epdl]. Secondary electrons are handled via simplified condensed-history transport or local energy deposition.

**RL environment.** A Gymnasium-compatible environment casts each photon history as an episode. Observations encode photon position, energy, direction, cross sections, and shell context. The hybrid action space combines a discrete interaction choice with continuous outputs for free path, scattering angles, energies, and secondary kinematics. The environment exposes energy-binned interaction histograms, per-process angular distributions, secondary summaries, and depth-dose diagnostics.

**Custom SAC stack.** Built on Stable-Baselines3 [@raffin2021stable] and PyTorch [@paszke2019pytorch], the SAC stack extends standard SAC with: a multi-head policy (discrete head, continuous heads, pretrained physics branch); $n$-step replay with bootstrapped critic targets; a multitask actor loss $\mathcal{L}_{\mathrm{actor}} = \mathcal{L}_{\mathrm{SAC}} + \lambda_{\mathrm{phys}}\mathcal{L}_{\mathrm{phys}}$; and hard-coded mean free paths from tabulated cross sections (a physics prior ensuring correct attenuation by construction).

**Training pipeline.** Training proceeds in two stages (\autoref{fig:training}). Stage I pretrains the physics branch on MC-generated labels using a composite supervised loss over interaction type, energy, angles, and secondary count. Stage II is a five-phase curriculum: Phases 0--1 train the discrete head with a log-likelihood-ratio reward under decaying teacher forcing; Phases 2--3 freeze the discrete head and train the continuous heads with kernel rewards (acceptance-probability matching, per-bin KL divergence, conservation penalties) under decaying teacher forcing; Phase 4+ refines remaining heads. During teacher-forced phases, the replay buffer stores MC-overridden actions, anchoring the critic to physics-consistent data.

![Two-stage training pipeline. Stage I pretrains the physics branch with supervised learning on MC labels. Stage II is a five-phase SAC curriculum: discrete interaction learning (Phases 0--1) followed by continuous kernel learning (Phases 2+), each with linearly decaying teacher forcing and phase-specific reward signals.\label{fig:training}](fig_training.pdf)

# Current limitations

The codebase is monolithic (a single Python script), the geometry is restricted to a homogeneous water phantom, angular distribution fine-tuning for continuous heads remains the principal unfinished component, and validation is at proof-of-concept level. Future work will modularise the code, expand material and geometry support, and complete angular convergence across all energy regimes.

# Research impact statement

Beam Weaver's immediate impact is as an openly archived, reproducible baseline for research on learned photon transport. The complete experimental workflow---MC data generation, physics-head pretraining, curriculum-based RL training, checkpointing, and process-level evaluation---is publicly inspectable, lowering the barrier for researchers who want to benchmark alternative policies, reward functions, or action parameterisations without rebuilding a transport-and-training pipeline from scratch. Even in its current state, the release captures a distinctive architecture, training regime, and transport-specific evaluation strategy that would otherwise remain unavailable. The Zenodo archive and public repository make the software citable, inspectable, and reusable [@beamweaver_software].

# AI usage disclosure

Generative AI tools (OpenAI ChatGPT and Anthropic Claude) were used to assist with code development, debugging, and manuscript preparation, including the architecture diagrams in this paper. The human authors defined the research question, designed all physics models and reward structures, made all architectural decisions, validated all code against known references, and remain fully responsible for the final submission.

# Acknowledgements

Beam Weaver builds on PENELOPE for transport methodology [@penelope2018], Geant4 for design inspiration [@agostinelli2003geant4], Stable-Baselines3 for RL infrastructure [@raffin2021stable], and PyTorch [@paszke2019pytorch]. It also draws on the RL literature on multi-step bootstrapping and entropy-regularised policy optimisation [@suttonbarto2018; @haarnoja2018sac], and on Hubbell and co-workers for atomic form factors and incoherent scattering functions [@hubbell1975].

# References
