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

Beam Weaver is an open-source Python framework for learning event-by-event photon transport in a homogeneous water phantom using a physics-informed hybrid Soft Actor-Critic (SAC) agent. It combines three tightly coupled components. A custom python Monte Carlo (MC) photon simulator using PENELOPE-derived cross-section tables for Rayleigh, Compton, and pair production [@penelope2018], and EPDL data for the photoelectric cross-sections [@cullen1997epdl, which can also perform condensed history transport for electrons. The second component is a Gymnasium-compatible reinforcement-learning (RL) environment that casts photon transport as a sequential decision problem, and the third consisting of a custom multi-head SAC that containes a pre-trainable physics head, which is coupled with a discrete head for interaction types and and a continuous head for continuous transport quantities---free path, scattering geometry, outgoing energy, and secondary-particle properties. Unlike standard SAC [@haarnoja2018sac], Beam Weaver uses $n$-step bootstrapped returns [@suttonbarto2018], a five-phase curriculum with scheduled teacher forcing, and physics-informed pretraining. \autoref{fig:architecture} shows the full actor-critic architecture.

![Beam Weaver actor-critic architecture. The observation vector feeds a shared residual MLP that branches into a discrete head (interaction type), continuous heads (angles, energies, secondaries), and a pretrained physics branch. The mean free path is hard-coded from tabulated cross sections. A curriculum wrapper selects the phase-appropriate reward, and teacher forcing overwrites replay-buffer actions during early phases.\label{fig:architecture}](fig_architecture.pdf)

# Statement of need

High-fidelity radiation transport relies on well-established MC codes such as PENELOPE, Geant4, and EGSnrc [@penelope2018; @agostinelli2003geant4; @kawrakow2000egsnrc], and this constitutes the state-of-the art in particle transport coding, where RL has not yet entered in full mod. Parallel to this, RL libraries like Stable-Baselines3 [@raffin2021stable] provide algorithmic infrastructure but no radiation-physics environments. Existing ML approaches to accelerating radiation transport focus on surrogate models for aggregate observables---dose maps, depth-dose curves, or MC denoising---rather than on learning the event-level transport process itself.

Beam Weaver addresses this question: can an RL agent learn to reproduce the stochastic sequence of interaction decisions that constitutes a photon history? The agent observes photon state variables and cross-section data, then predicts the next interaction class together with continuous outcomes that can be compared against MC references process by process. The great advantage is precisely this one, MC can produce an arbitrarily large number of observables to train the agent, making it an excellent candidate for RL approaches.

The release bundles the complete workflow: MC data generation, physics-head pretraining, curriculum-based SAC training, checkpointing, and evaluation through interaction statistics, secondary-particle summaries, tracks, and percentage depth-dose curves---making it an inspectable research pipeline, not merely a model checkpoint.

# State of the field

Established MC codes provide validated photon transport across broad material and geometry classes [@penelope2018; @agostinelli2003geant4; @kawrakow2000egsnrc]. In its current formulation, Beam Weaver  uses a deliberately narrow setting---a $10 \times 10$~cm$^2$ beam on a $100 \times 100 \times 100$~cm$^3$ water phantom, to study whether a SAC agent can learn event-level transport from a physics-based simulator.

SAC was chosen because its maximum-entropy formulation produces inherently stochastic policies (physically appropriate when multiple interactions carry non-zero probability), it is off-policy (essential for learning from teacher-forced replay buffers), it handles continuous action spaces natively, and its automatic entropy tuning adapts exploration across curriculum phases with different natural entropy scales.

# Software design

Beam Weaver is organised around three components, currently distributed as a single Python research script with accompanying physics tables and metadata.

**MC reference simulator.** A compact Python simulator samples Rayleigh, Compton, photoelectric, and pair-production interactions using PENELOPE and EPDL-derived cross sections [@penelope2018], [@cullen1997epdl], Hubbell incoherent scattering functions and coherent form factors [@hubbell1975]. Secondary electrons are handled via simplified condensed-history transport or local energy deposition.

**RL environment.** A Gymnasium-compatible environment casts each photon history as an episode. Observations encode photon position, energy, direction, cross sections, and shell context. The hybrid action space combines a discrete interaction choice with continuous outputs for free path, scattering angles, energies, and secondary kinematics. The environment exposes energy-binned interaction histograms, per-process angular distributions, secondary summaries, and depth-dose diagnostics.

**Custom SAC stack.** Built on Stable-Baselines3 [@raffin2021stable] and PyTorch [@paszke2019pytorch], the SAC stack extends standard SAC with: a multi-head policy (discrete head, continuous heads, pretrained physics branch); $n$-step replay with bootstrapped critic targets; a multitask actor loss $\mathcal{L}_{\mathrm{actor}} = \mathcal{L}_{\mathrm{SAC}} + \lambda_{\mathrm{phys}}\mathcal{L}_{\mathrm{phys}}$; and hard-coded mean free paths from tabulated cross sections (a physics prior ensuring correct attenuation by construction).

**Training pipeline.** Training proceeds in two stages (\autoref{fig:training}). Stage I pretrains the physics branch on MC-generated labels using a composite supervised loss over interaction type, energy, angles, and secondary count. Stage II is a five-phase curriculum: Phases 0--1 train the discrete head with a log-likelihood-ratio reward under decaying teacher forcing; Phases 2--3 freeze the discrete head and train the continuous heads with kernel rewards (acceptance-probability matching, per-bin KL divergence, conservation penalties) under decaying teacher forcing; Phase 4+ refines remaining heads. During teacher-forced phases, the replay buffer stores MC-overridden actions, anchoring the critic to physics-consistent data.

![Two-stage training pipeline. Stage I pretrains the physics branch with supervised learning on MC labels. Stage II is a five-phase SAC curriculum: discrete interaction learning (Phases 0--1) followed by continuous kernel learning (Phases 2+), each with linearly decaying teacher forcing and phase-specific reward signals.\label{fig:training}](fig_training.pdf)

# Current limitations

The codebase is monolithic (a single Python script), the geometry is restricted to a homogeneous water phantom, angular distribution fine-tuning for continuous heads remains the principal unfinished component, and validation is at proof-of-concept level. Future work will modularise the code, expand material and geometry support, and complete angular convergence across all energy regimes.

# Research impact statement

Beam Weaver's immediate impact is as an openly archived, reproducible baseline for research on learned photon transport. The complete experimental workflow---MC data generation, physics-head pretraining, curriculum-based RL training, checkpointing, and process-level evaluation---is publicly inspectable, lowering the barrier for researchers who want to benchmark alternative policies, reward functions, or action parameterisations without rebuilding a transport-and-training pipeline from scratch. Even in its current state, the release captures a distinctive architecture, training regime, and transport-specific evaluation strategy that would otherwise remain unavailable. The Zenodo archive and public repository make the software citable, inspectable, and reusable [@beamweaver_software].

# Preliminary results

The results presented here are from a partially trained agent
evaluated at 1 MeV incident photon energy on a 100 × 100 × 100 cm³
water phantom. At the time of evaluation, the agent had completed the
full discrete interaction curriculum (Phases 0--1) and was partway
through the continuous kernel curriculum (Phase 2, energy regime 2 of
16, covering 5--10 keV). The continuous transport heads had therefore
received supervised angular training only in the low-energy
photoelectric-dominated regime; all behaviour at higher energies
reflects the physics-seeded initialization plus the RL reward signal,
without direct angular curriculum exposure. These results should
accordingly be interpreted as a progress snapshot rather than a
converged validation.

The evaluation compared 10 000 photon histories generated by the
internal MC reference simulator against 10 000 histories generated by
the agent, both using condensed-history electron transport for
secondary energy deposition. Table 1 summarizes the key metrics.

**Table 1.** Evaluation summary at 1 MeV (10 000 photon histories).

| Metric                          |       MC |    Agent |    Ratio |
|:--------------------------------|---------:|---------:|---------:|
| Total interactions              |  137 701 |   66 562 |     0.48 |
| Mean track length (interactions)|     13.8 |      6.7 |     0.48 |
| Total dose deposited (MeV)     | 10 689.7 |  8 823.1 |    0.825 |
| Normalised PDD L₂ distance     |      --- |      --- |   0.226  |
| Compton fraction                |    0.922 |    0.886 |     0.96 |
| Rayleigh fraction               |    0.033 |    0.025 |     0.74 |
| Photoelectric fraction          |    0.044 |    0.089 |     2.00 |
| Compton mean angle (°)          |     64.7 |     53.5 |      --- |
| Rayleigh mean angle (°)         |      7.2 |     53.0 |      --- |
| Simulation time (s)             |    464.4 |    151.5 |   3.1×   |

## Physics enforcement mechanisms

Before discussing the evaluation results, it is worth summarizing the
structural constraints that are hard-wired into the agent's action
interpretation, as distinct from the soft reward signals described in
the reward design section. These constraints ensure that certain
physical conservation laws hold exactly in every agent-generated event,
regardless of what the policy network outputs.

**Energy conservation.** The agent's continuous energy outputs are not
interpreted as absolute energies. Instead, they are treated as relative
weights that define a fractional partition of the available kinetic
energy. For Compton scattering, the two energy outputs (scattered
photon and recoil electron) are normalized so that
$E_\gamma^{\mathrm{out}} + E_e = E_\gamma^{\mathrm{in}}$ exactly. For
pair production, the two secondary energy outputs are similarly
normalized so that
$E_{e^-} + E_{e^+} = E_\gamma^{\mathrm{in}} - 2m_e c^2$ exactly. For
photoelectric absorption, the electron receives the full available
energy $E_e = E_\gamma^{\mathrm{in}} - E_b$ after shell-specific
binding energy subtraction, and the scattered photon energy is forced
to zero. For Rayleigh scattering (elastic), the outgoing photon energy
equals the incoming energy by construction. These fractional-split
constraints mean that the agent cannot violate energy conservation
regardless of its network outputs; it can only control *how* the
available energy is distributed among outgoing particles.

**Secondary particle multiplicity.** The number of secondary electrons
is structurally determined by the interaction type rather than freely
predicted by the agent. Rayleigh scattering produces no secondaries.
Compton scattering and photoelectric absorption each produce exactly
one electron. Pair production produces exactly two charged particles
(electron and positron). This structural masking is enforced through
a dedicated masking function which zeros out unused secondary-particle
action slots based on the discrete interaction choice. The agent
therefore cannot produce unphysical secondary counts (for example, two
electrons from a Compton event or zero electrons from photoelectric
absorption).

**Forbidden photon energy.** For photoelectric absorption and pair
production---interactions where the photon is fully absorbed---the
agent's scattered-photon energy output is overridden to zero. If the
agent's raw energy output is nonzero for these interaction types, a
hard penalty of $r_{E\text{-corr}} = -5$ is applied in addition to the
override, providing both a hard constraint and a learning signal.

**Shell-specific binding energies.** For photoelectric absorption, the
binding energy subtracted from the available kinetic energy is
determined by the specific atomic shell selected by the MC reference
sampler (H K, O K, O L1, O L2, or O L3), using shell-resolved
cross-section tables for water. The shell identity is communicated to
the agent through a one-hot observation vector, ensuring that the
energy balance accounts for the correct shell-specific Q-value.

**Pair production threshold.** The discrete head's logits for pair
production are suppressed by $-10^9$ whenever the photon energy falls
below $2m_e c^2 = 1.022$ MeV, preventing the agent from selecting a
kinematically forbidden interaction.

**Mean free path from physics.** The mean free path is not a learned
quantity in the current architecture. It is computed from the tabulated
total attenuation coefficient $\mu(E)$ and sampled from the
corresponding exponential distribution $p(s) = \mu e^{-\mu s}$, with
the $\mu$ output frozen to its physics-derived value throughout
training. This ensures that the spatial distribution of interaction
sites matches the reference transport regardless of other policy
outputs.

These hard constraints collectively guarantee that every agent-generated
event satisfies energy conservation, produces the correct number of
secondary particles for the selected interaction type, respects
kinematic thresholds, and samples interaction sites from the correct
spatial distribution. The remaining degrees of freedom---interaction
type selection, scattering angles, energy partition ratios, and
secondary-particle directions---are learned through the reward signal
and constitute the quantities evaluated below.

## Interaction type selection

The discrete head reproduces the Compton-dominated interaction mixture
at 1 MeV with high accuracy: the Compton fraction is within 4% of the
MC reference. Rayleigh and photoelectric fractions show larger
deviations---Rayleigh is under-predicted by a factor of 0.74 and
photoelectric is over-predicted by a factor of 2---but these are
minority channels at 1 MeV (together accounting for less than 8% of MC
interactions) and the agent has not yet undergone angular curriculum
training at energies where these channels dominate. The interaction
proportions are shown in Figure 1 (left panel).

## Depth--dose behaviour

Figure 1 (right panel) compares the percentage depth--dose (PDD)
curves. The agent reproduces the overall exponential attenuation
profile, with good shape agreement in the first 30--40 cm of depth.
Beyond approximately 50 cm, the agent's PDD falls off more steeply
than the MC reference, resulting in a total deposited dose that is
82.5% of the MC value. This under-deposition at depth is consistent
with the shorter mean track length (6.7 versus 13.8 interactions per
photon): incorrect angular distributions cause unphysical energy
transfers that terminate photon histories prematurely. Despite
the hard energy conservation constraint, incorrect energy *partition
ratios* (driven by the uncoupled angle--energy parameterization
discussed below) lead to excessive energy transfer to secondary
electrons at individual interaction sites, reducing the photon energy
below the transport cutoff sooner than in the MC reference.

![Interaction type proportions (left) and percentage depth--dose
comparison (right) for 10 000 photon histories at 1 MeV. The agent
reproduces the Compton-dominated interaction mixture and the overall
PDD shape, with increasing divergence at depth beyond 50 cm due to
untrained angular distributions.](fig3_interactions_tracks.pdf){#fig:interactions width=48%}
![](fig1_pdd_comparison.pdf){#fig:pdd width=48%}

## Angular distributions

Figure 2 shows the scattering-angle distributions for Rayleigh,
Compton, and photoelectric interactions. Rayleigh scattering in the MC
reference is tightly forward-peaked (mean 7.2°, 95th percentile
18.9°), as expected from the coherent form-factor weighting at 1 MeV.
The agent instead produces a broad angular distribution (mean 53.0°)
that spans the full 0--180° range. Compton scattering shows
qualitatively better agreement (MC mean 64.7° versus agent mean 53.5°),
but with substantially higher variance (agent standard deviation 61.4°
versus MC 42.3°) and a spurious concentration of events near 0°. The
photoelectric electron angular distribution is consistent between MC
and agent, both showing broad distributions as expected from the
Sauter and isotropic angular models used for K-shell and L-shell
ejection respectively.

These angular discrepancies have a known cause: the current
implementation parameterizes scattering angles by mapping the policy
network's tanh-squashed output linearly to $\theta \in [0, \pi]$. The
tanh nonlinearity concentrates probability density near the midrange
($\theta \approx 90°$) and exponentially suppresses the tails, making
forward-peaked distributions structurally difficult to represent
regardless of the reward signal. A reparameterization from $\theta$ to
$\cos\theta$ is planned for the next release.

![Scattering-angle distributions for Rayleigh (left), Compton
(centre), and photoelectric (right) interactions. The MC reference
shows the expected forward-peaked Rayleigh distribution and broad
Compton distribution; the agent's angular distributions are broadened
by a tanh parameterization artifact discussed in the
text.](fig2_angular_distributions.pdf){#fig:angles width=100%}

## Compton energy--angle consistency

Figure 3 examines the kinematic consistency of Compton scattering
events by plotting the scattered photon energy against scattering angle
alongside the Klein--Nishina prediction. In the MC reference, all
events lie on or near the analytic curve
$E_{\mathrm{out}} = E_{\mathrm{in}} / (1 + \alpha(1 - \cos\theta))$.
The agent's events form a diffuse cloud with weak correlation between
angle and energy.

This occurs because the current architecture independently predicts the
energy partition ratio and scattering angle for Compton events. While
energy conservation is guaranteed by the fractional-split constraint
($E_\gamma^{\mathrm{out}} + E_e = E_\gamma^{\mathrm{in}}$ exactly),
the *ratio* of that split is a free parameter that should be
deterministically related to the scattering angle through the Compton
formula. The agent's two continuous outputs (energy fraction and photon
angle) are coupled only through a soft kinematic consistency penalty
($\lambda = 0.05$) and the acceptance kernel reward. The combined
constraint is not yet sufficient to enforce the tight angle--energy
correlation required by Klein--Nishina kinematics. A fourfold increase
in the consistency penalty coefficient ($\lambda = 0.05 \to 0.20$) is
planned for the next training iteration.

![Compton scattering angle versus scattered photon energy for MC
(left) and agent (right). The dashed line shows the Klein--Nishina
prediction at 1 MeV. MC events follow the analytic curve; agent events
show weak angle--energy correlation. Note that energy conservation
holds exactly in both cases (the photon and electron energies sum to
$E_\gamma^{\mathrm{in}}$); the discrepancy is in the *partition
ratio*, which is not yet constrained to follow the Compton
formula.](fig6_compton_angle_energy.pdf){#fig:compton_ae width=100%}

## Secondary particle properties

Table 2 summarizes the secondary electron properties. As noted above,
the number of secondaries per interaction is structurally enforced:
every Compton event produces exactly one recoil electron, every
photoelectric event produces exactly one ejected electron, and every
pair production event produces one electron and one positron. The
counts in Table 2 therefore reflect the total number of each
interaction type rather than a learned multiplicity.

The Compton recoil electron count is proportional to the total number
of Compton events (agent produces 58 996 versus MC 126 993, reflecting
the shorter track lengths). The mean Compton electron energy is higher
for the agent (0.136 MeV versus 0.069 MeV), consistent with the energy
partition inconsistency discussed above: the agent sometimes assigns
excessive energy to the electron at small scattering angles, which is
kinematically forbidden under Klein--Nishina but permitted by the
current soft constraint.

For photoelectric electrons, the agent produces correct counts (5 600
O K-shell electrons versus MC 5 799, and 313 O L1-shell electrons
matching MC exactly) but substantially lower mean energies (agent
0.009 MeV versus MC 0.050 MeV for O K-shell). This discrepancy arises
because the agent has only trained through the 5--10 keV energy regime,
where photoelectric electron energies are indeed on the order of a few
keV. At the 1 MeV evaluation energy, the available energy after K-shell
binding subtraction is approximately 0.999 MeV, but the agent's energy
partition---seeded from the low-energy regime---has not yet adapted to
higher energies. This is expected to resolve as the curriculum
progresses through higher-energy regimes.

**Table 2.** Secondary electron statistics.

| Label       | MC count | MC mean E (MeV) | Agent count | Agent mean E (MeV) |
|:------------|--------:|-----------------:|------------:|-------------------:|
| compton\_e  | 126 993 |           0.069  |      58 996 |             0.136  |
| photo\_O\_K |   5 799 |           0.050  |       5 600 |             0.009  |
| photo\_O\_L1|     313 |           0.051  |         313 |             0.009  |

## Free path distributions

Free path distributions for Compton scattering agree well between MC
and agent (MC mean 7.6 cm, agent mean 8.8 cm). As noted above, the
mean free path is computed from the tabulated attenuation coefficient
rather than learned, so this agreement reflects the physics
initialization rather than training convergence. Photoelectric free
paths are shorter for the agent (1.6 cm versus 4.0 cm), consistent
with the over-representation of photoelectric events at energies where
Compton should dominate.

## Spatial fluence

Figure 4 shows the XZ fluence maps for MC and agent showers. Both
exhibit the expected forward-peaked pencil-beam structure with lateral
scatter increasing with depth. The agent's fluence is truncated at
shallower depths, consistent with the shorter track lengths and
steeper PDD falloff discussed above. The lateral spread at matched
depths is qualitatively similar, suggesting that the overall transport
geometry is reasonable even though the per-event angular distributions
are not yet converged.

![XZ spatial fluence maps for MC (left) and agent (right), showing
photon interaction positions projected onto the depth--lateral plane.
Both show forward-peaked structure with lateral spreading; the agent's
fluence is truncated at depth due to premature track
termination.](fig8_spatial_fluence.pdf){#fig:fluence width=100%}

## Computational performance

The agent generated 10 000 photon histories in 151.5 seconds compared
to 464.4 seconds for the MC reference, a speedup of approximately
3.1×. This comparison should be interpreted with caution: neither
implementation is optimized for throughput, the MC simulator is written
in pure Python without vectorization, and the agent's inference
includes per-photon GPU transfers that could be amortized through
batching. The speedup is reported here only to establish that the
agent's inference cost is in the same order as the reference, not to
claim a definitive performance advantage.

## Identified improvements for the next release

The evaluation identifies four concrete improvements planned for the
immediate next training iteration:

1. **Angle reparameterization.** The continuous action slots for polar
   angles will be changed from $\theta \in [0, \pi]$ to
   $\cos\theta \in [-1, 1]$. This places the bulk of physical
   scattering distributions (Klein--Nishina, Rayleigh form factor,
   Sauter) in the linear region of the tanh activation, where gradient
   flow is strongest, and eliminates the structural suppression of
   forward-peaked and backward-peaked angular distributions.

2. **Stronger kinematic consistency.** The Compton electron consistency
   penalty coefficient will be increased from $\lambda = 0.05$ to
   $\lambda = 0.20$ to enforce tighter coupling between the predicted
   photon angle and the implied recoil electron kinematics.

3. **Energy regime truncation.** Episodes will be terminated when the
   photon energy drops below the current curriculum regime floor during
   Phase 2 training. This prevents out-of-regime transitions---where
   the continuous head is untrained---from entering the replay buffer
   and contaminating the critic's value estimates for energy ranges
   that will be covered by later curriculum stages.

4. **Feature extractor freezing.** The shared feature extractor will be
   frozen during Phase 2 and beyond to prevent representation drift
   from degrading the pretrained physics heads. Training diagnostics
   show that physics head losses increase during Phase 2 as the RL
   actor loss reshapes the shared feature representation away from the
   physics-optimal configuration established during pretraining.


# AI usage disclosure

Generative AI tools (OpenAI ChatGPT and Anthropic Claude) were used to assist with code development, debugging, and manuscript preparation, including the architecture diagrams in this paper. The human authors defined the research question, designed all physics models and reward structures, made all architectural decisions, validated all code against known references, and remain fully responsible for the final submission.

# Acknowledgements

Beam Weaver builds on PENELOPE for transport methodology [@penelope2018], Geant4 for design inspiration [@agostinelli2003geant4], Stable-Baselines3 for RL infrastructure [@raffin2021stable], and PyTorch [@paszke2019pytorch]. It also draws on the RL literature on multi-step bootstrapping and entropy-regularised policy optimisation [@suttonbarto2018; @haarnoja2018sac], and on Hubbell and co-workers for atomic form factors and incoherent scattering functions [@hubbell1975].

# References
