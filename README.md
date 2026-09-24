# Beam Weaver

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18994134.svg)](https://doi.org/10.5281/zenodo.18994134)

**Beam Weaver is a learned-kernel Monte Carlo framework for photon transport in water.**

## Introduction

Beam Weaver is a proof-of-concept neural network that can be taught how to simulate photon/electron transport, and later infer practically all stochastic quantities in an explicit, recursive manner similar to Monte Carlo.

The idea behind this proof-of-concept is to explore the possibilities of using a neural network framework in Monte Carlo radiation transport, which may eventually lead to improvements in dosimetric calculations. The neural network framework explored by Beam Weaver entails the explicit transport of particles (for now, limited to photon/electron/positron transport) rather than the more common approaches of implicit dosimetric calculations. This architectural choice was deliberate, the idea is to answer the very simple question "can a neural network learn how to transport particles?". As such, Monte Carlo sampling is the ideal teacher, given it can produce an arbitrarily large pool of stochastic outcomes that can be fed into the neural network. Somewhat similar approaches have been implemented by other authors [1,2].

Initially, the developed architecture involved  a Soft Actor–Critic agent, but training revealed that the previously supervised physics head used to kickstart the agent already accurately reproduced the photon transport, generating complete recursive showers without the need for reinforcement updates.  This shifted the focus from a reward-driven reinforcement learning agent to a direct framework, learning the local stochastic transition kernels directly from the Monte Carlo sampling. The mathematical framework also favoured this approach [3]. After the teaching process, Beam Weaver’s photon-electron showers are generated recursively just like Monte Carlo, strictly from values inferred by the  neural network heads, while imposing deterministic physics conservation laws. Beam Weaver was trained using *Beam Spinner*, a self-developed Python Monte Carlo code for photon and electron transport in liquid water (1 keV–10 MeV), largely inspired by the PENELOPE code [4]. Beam Spinner can sample photon free paths, Rayleigh, Compton, photoelectric and pair events, using EPDL cross-sections and tabular coherent and incoherent scattering functions [5, 6]; and simulate electron CSDA transport using ESTAR stopping power tables [7]. 

After training, In its current form, Beam Weaver can successfully transport photons in a liquid water phantom with energies ranging from 1 keV to 10 MeV. Practically all stochastic quantities are inferred (interaction choice, photon shells in the case of photoelectric events, scattering polar and azimuth photon and electron angles, shared kinetic energy in pair production), totalling 13 inferred quantities, each with its own head, while exponential free path sampling and electron transport still use Beam Spinner’s classical Monte Carlo approach.

## Mathematical framework

Each stochastic quantity (interaction type, atomic subshell, angle, etc.) is given its own disjoint trainable categorical head.
Discrete quantities, such as the interaction process and atomic subshell, are inherently categorical therefore directly represented as such by their respective heads. 
Continuous quantities,such as angular variables and energy fractions, must be divided into bins. Binning takes into account the natural variables that define each quantity. 

### Factorized interaction law

```math
\pi_{\Theta}(\mathbf{Z}_t\mid E_t)
=
\prod_{h=1}^{13}
q_{\theta_h}(Z_{t,h}\mid X_{t,h})^{m_{t,h}},
\qquad
m_{t,h}\in\{0,1\}.
```

At interaction $t$, the probability Beam Weaver assigns to the respective category/bin label
$\mathbf{Z_t}$ is the product of all the probabilities, predicted by each and every active head, of their respective outcomes $Z_{t,h}$, given the incident photon energy $E_t$; and the atomic shell, the sampled pair-production energy fraction, when necessary. 

The binary mask $m_{t,h}$ serves to activate the heads $h$ applicable to the interaction. It's a simple binary value, when $m_{t,h}=1$,  $q^1=q$. When $m_{t,h}=0$, $q^0=1$.

| Symbol | Meaning |
| :--- | :--- |
| $\pi_{\Theta}(\mathbf{Z_t}\mid E_t)$ | Final probability assigned by Beam Weaver to the interaction category and bin labels $\mathbf{Z}_t$, given the incoming photon energy $E_t$. |
| $\Theta=\{\theta_h\}_{h=1}^{13}$ | Collection of trainable parameter sets for the 13 heads. |
| $t$ | Interaction index. |
| $h$ | Head index, from 1 to 13. |
| $E_t$ | Photon energy immediately before interaction $t$. |
| $\mathbf{Z_t}$ | Categorical representation of each and every stochastic quantities for interaction $t$. |
| $Z_{t,h}$ | Categorical representation of the stochastic quantity assigned to head $h$, at interaction $t$, when that head is active. |
| $X_{t,h}$ | Conditioning input supplied to head $h$: the photon energy and any required earlier outcomes within the same interaction event. |
| $\theta_h$ | Trainable parameters of head $h$. |
| $q_{\theta_h}(\cdot\mid X_{t,h})$ | Conditional probability distribution predicted by head $h$ over its output bins. |
| $q_{\theta_h}(Z_{t,h}\mid X_{t,h})$ | Predicted probability of the particular output bin $Z_{t,h}$. |
| $m_{t,h}$ | Binary applicability mask: $1$ when head $h$ is active for the event and $0$ otherwise. |



### Masked cross-entropy training

For each stochastic quantity, Beam Spinner supplies the observed category or bin label from the simulated interactions, which are used as reference targets to train the corresponding heads. 

The training objective is

```math
\mathcal{L}(\Theta)
=
\sum_{h=1}^{13}
\sum_{g\in\mathcal{G}_h}
H\left(
\widehat{\mathbf{p}}_{g,h},
\mathbf{q}_{\theta_h}(\cdot\mid X_{g,h})
\right).
```

Here, cross-entropy is defined by

```math
H(\mathbf{p},\mathbf{q})
=
-\sum_{k=1}^{K_h} p_k \ln q_k
=
H(\mathbf{p})
+
D_{\mathrm{KL}}(\mathbf{p}\Vert\mathbf{q}).
```


The set $\mathcal{G}_h$ contains only the targets for which head $h$ is active; inactive heads therefore do not contribute to the corresponding training loss.

At the event level, taking the negative logarithm of the factorized event probability produces a sum over active heads, this provides a very convenient framework as it allows each head to be trained separately. For a fixed target distribution, its entropy does not depend on the model parameters, so minimizing cross-entropy also minimizes KL divergence [3].

| Symbol | Meaning |
| :--- | :--- |
| $\mathcal{L}(\Theta)$ | Total training loss. |
| $\Theta$ | Collection of trainable parameters across all 13 heads. |
| $h$ | Head index, from 1 to 13. |
| $\theta_h$ | Trainable parameters of head $h$. |
| $\mathcal{G}_h$ | Set of training-target indices for which head $h$ is active. |
| $g$ | Index identifying an individual event or a reference sampling set (one sampled quantity at one fixed incident photon energy, plus a fixed shell where relevant). |
| $X_{g,h}$ | Conditioning input supplied to head $h$ for target $g$. |
| $\widehat{\mathbf{p}}_{g,h}$ | Beam Spinner target probability vector: empirical bin frequencies for a reference sampling set, or a one-hot vector for an individual event. |
| $\mathbf{q_{\theta_h}} (\cdot \mid X _{g,h} )$ | Probability vector predicted by head $h$ for the given conditioning input. |
| $K_h$ | Number of output bins for head $h$. |
| $k$ | Output-bin index, from 1 to $K_h$. |
| $p_k,\ q_k$ | Components of the target and predicted probability vectors in the definition of cross-entropy. |
| $H(\mathbf{p},\mathbf{q})$ | Cross-entropy between the target and predicted distributions. |
| $H(\mathbf{p})$ | Entropy of the target distribution, constant with respect to the model parameters. |
| $D_{\mathrm{KL}}(\mathbf{p}\Vert\mathbf{q})$ | Kullback–Leibler divergence from the target distribution to the predicted distribution. |

## Method

Beam Spinner recursively generates individual Monte Carlo samples for each interaction
process at a specified photon energy,  and, where applicable,
for each atomic subshell.  Beam Weaver uses these samples to learn  outcome probabilities  through its categorical output heads. 

Beam Weaver contains 13 disjoint heads, 11 two-hidden-layer 64-unit SiLU MLPs, and 2 learned 36-logit azimuth vectors (given that, for unpolarized photon scattering in an isotropic medium there is no preferred azimuthal direction, which therefore defines a uniform target distribution, no energy or polar angle input is necessary, the logits are directly compared to the Monte Carlo sampled azimuth frequencies). This totals 155,127 trainable parameters, if default initial binning is used. A softmax converts each head's logits into probabilities.

| Head $h$ | Stochastic quantity | Categories or represented quantity | Physical inputs | Default categories or bins $K_h$ (binning resolution can be increased if sampling is sparse) |
| :---: | :--- | :--- | :--- | ---: |
| 1 | Interaction process | Rayleigh, Compton, photoelectric absorption, pair production | $E$ | 4 |
| 2 | Rayleigh polar scattering angle | $s_R=\ln[(1-\cos\theta_R)/2]$ | $E$ | 720 |
| 3 | Rayleigh azimuthal angle | $\phi_R$ | None | 36 |
| 4 | Compton normalized energy transfer | $u=(1-\tau)/(1-\tau_{\min})$ | $E$ | 180 |
| 5 | Compton azimuthal angle | $\phi_C$ | None | 36 |
| 6 | Photoelectric subshell | H-K, O-K, O-L1, O-L2, O-L3 | $E$ | 5 |
| 7 | Photoelectron polar emission angle | $\nu_{Ph}=1-\cos\theta_{Ph}$ | $E$, selected subshell | 180 |
| 8 | Photoelectron azimuthal angle | $\phi_{Ph}$ | Selected subshell | 36 |
| 9 | Pair-production kinetic-energy sharing | $f=T_-/(T_-+T_+)$ | $E$ | 90 |
| 10 | Pair electron polar emission angle | $\nu_{pp}^{-}=1-\cos\theta_{pp}^{-}$ | $E,\ f$ | 180 |
| 11 | Pair electron azimuthal angle | $\phi_{pp}^{-}$ | $E,\ f$ | 36 |
| 12 | Pair positron polar emission angle | $\nu_{pp}^{+}=1-\cos\theta_{pp}^{+}$ | $E,\ 1-f$ | 180 |
| 13 | Pair positron azimuthal angle | $\phi_{pp}^{+}$ | $E,\ 1-f$ | 36 |

Both polar and azimuthal angles are defined in local frames with the polar axis following the incident photon direction, and later rotated to the lab frame if necessary for transport. Energies are represented in a logarithmic scale; also, subshell inputs use a five-component one-hot representation [(1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1)]. Finally, $E$ and $E'$ are the incident and scattered photon energies; and for Compton, $\tau=E'/E$, and $\tau_{\min}=1/(1+2E/(m_ec^2))$, where $m_ec^2$
is the electron rest energy. The quantities $T_-$ and $T_+$ are the electron and positron kinetic energies in pair production.

Each head learns a categorical distribution; continuous variables are sampled within the selected bin and transformed back to physical quantities using the appropriate constants.

For continuous quantities other than azimuths, Monte Carlo sampling outcomes are pooled with equal contributions from each relevant training energy and, where applicable, subshell. Bin boundaries are placed at cleverly chosen quantiles: intervals are narrower where the pooled distribution is concentrated and wider where it is sparse, giving approximately equal pooled occupancies. Azimuthal angles use 36 equal-width bins over $[0,2\pi)$. Before training, the binning resolution is checked against validation samples to assess the uniform-within-bin approximation. For a quantity that fails this check, the number of bins is doubled once and the boundaries are recalculated from the training samples. The final binning remains fixed during training and transport.

For nine heads, the reference targets are normalized category or bin counts, $\hat{p_{g,h,k}}=n_{g,h,k}/N_{g,h}$, where $n_{g,h,k}$ counts samples in category or bin $k$ and $N_{g,h}$ is the total count for that target. The four pair-direction heads instead use individual interaction samples, retaining the continuous electron/positron energy fraction as an input. Their targets are one-hot vectors: $1$ for the observed angular bin and $0$ elsewhere.

Training minimizes cross-entropy against reference samples or their empirical category distributions. The heads use photon energy and, where required, the selected shell or sampled electron/positron energy fraction. Training proceeds head by head, with validation-based stopping and restoration of the best weights when overfitting. The learned energy domain is **0.001–10 MeV**.

During transport, the predicted probabilities are used to sample the interaction process and its associated quantities. For a continuous quantity, a bin is selected first and a value is then sampled uniformly within that interval in the represented quantity listed above. This allows the inferred stochastic quantities to display continuous behaviour - using only bin centres would restrict the sampled angles and energy fractions to a fixed set of values. Inverse transformations and the implemented kinematic relations yield the physical energies and directions. Uniform sampling within each bin is the remaining approximation to the distribution inside that interval.

## Experiment

After Beam Spinner taught Beam Weaver; 50,000 monodirectional and monochromatic photon histories were generated within a 10 × 10 cm² square and transported through a 100 × 100 × 100 cm³ water phantom at five different initial energies (0.1, 1, 2, 5, and 10 MeV). Two Beam Spinner runs were performed at different PRNG seeds, and one run using Beam Weaver alone. The idea was to demonstrate the fidelity of Beam Weaver’s transport capabilities. Beam Spinner and Beam Weaver’s runs shared the mean free path Simulator and the electron/positron condensed history transport. All other quantities were inferred by Beam Weaver with only prior knowledge of the energy and direction of the source particles, generating the rest recursively. Results presented here cover all five initial photon energies. Error bars are not shown for enhanced visuals.

## Results

[Results gallery](results/README.md) presents the **v0.4.0** results at **0.1, 1, 2, 5 and 10 MeV**: 50,000 primary photons per simulation, two independent Beam Spinner runs, and one Beam Weaver run. Available figures cover depth dose, Compton and photoelectric angles, shell selection, pair kinetic-energy sharing and interaction fractions where the saved data support them.

![5 MeV depth-dose comparison](results/figures/5MeV/pdd.png)



## Interpretation and limitations

For the time being, Beam Weaver's geometry has been deliberately made very simple, with simple monochromatic, monodirectional 10x10 $\mathbf{cm^2}$ beams impinging on a 100x100x100 $\mathbf{cm^3}$ water phantom. For dose tallying, 100, 1 cm thick, depth splabs are used. Beam Weaver and Beam Spinner tally deposited energy, rather than absorbed dose.

Beam Spinner's MC1–MC2 difference is used to illustrate sampling variability and how it compares, even if qualitatively, to Beam Weaver's inferrence.

As previously mentioned, free paths and electron transport (and actual energy deposition) are still computed with conventional Monte Carlo.

Future steps include comparison against PENELOPE, implementing electron/positron transport dedicated heads, variance reduction techniques, and inferrence speed-up algorithms. Beam Weaver remains practical as a proof-of-concept and does not claim to improve sampling at the moment.



## Installation

Requires **Python 3.10 or later**. Clone the repository and install:

```bash
git clone https://github.com/PedroTelesFCUP/Beam_weaver.git
cd Beam_weaver
python -m pip install -e .
```

The base installation includes NumPy, pandas and Matplotlib. Training, learned transport and learned-head validation also require PyTorch:

```bash
python -m pip install -e '.[learn]'
```

For GPU use, install the appropriate PyTorch build for your machine. Commands accept `--device cpu`, `--device cuda:0`, or the default `--device auto`.

Start the interactive program with:

```bash
python -m beamweaver
```

The installed `beamweaver` command and `python Beam_weaver_0_4_0.py` open the same application. The launcher must remain beside the `beamweaver` package when used without installation.

| Option | Operation |
| --- | --- |
| 1 | Generate training/validation/test data |
| 2 | Train all heads or one selected head |
| 3 | Run an audited shower |
| 4 | Compare MC1, MC2 and BeamWeaver |
| 5 | Validate factors or reference samplers |
| 6 | Exit |

Each operation explains its purpose and default sampling or training choices before asking for inputs. Help and the menu can be opened without physics tables or a checkpoint. By default, each operation creates a new directory such as `runs/generate-20260923-v1`; the next run of that operation on the same UTC date uses `v2`, then `v3`. These suffixes count runs **for that day**, not Beam Weaver software versions. An explicit `--output` chooses a directory instead; `--resume` continues the specified training run.

## Water data

The six water CSV tables are included in the repository root. Commands below use `--data-dir .` when run from that directory. For tables stored elsewhere, supply their directory instead. The program reads these filenames and columns:

| File | Columns used |
| --- | --- |
| `Final_cross_sections.csv` | `E`, `photoelectric`, `compton`, `pair_triplet` |
| `Rayleigh_cross_sections.csv` | `E`, `coh` |
| `WaterPhotoShells.csv` | `E_MeV`, `H_K_cm2g`, `O_K_cm2g`, `O_L1_cm2g`, `O_L2_cm2g`, `O_L3_cm2g` |
| `water_fq.csv` | `q`, `F_q` |
| `water_sq.csv` | `q`, `S_q` |
| `ElectronStoppingPower.csv` | `E_MeV`, `S_col_MeV_per_cm`, `S_rad_MeV_per_cm` |

Generation and validation require the first five files. A learned shower requires the first four plus stopping powers; comparison requires all six. Training reads the generated dataset and does not need the CSV tables. Synthetic tables under `tests` are regression fixtures, not water reference data. See [water-table provenance and conventions](docs/water-data.md).

## Generation of Monte-Carlo samples, training, validation, and comparison

Commands support the same workflow as the menu. Use a new or empty output directory; omit `--output` to create the next available `runs/<command>-YYYYMMDD-vN` directory (UTC date, sequential run number). Dataset and checkpoint basenames are unchanged; command audit records also use a dated `command-YYYYMMDD-vN.json` name.

```bash
python -m beamweaver generate --data-dir . --output runs/data
python -m beamweaver train runs/data/schema_v4_data.npz --output runs/training
```

### Beam Spinner sampling

Beam Spinner generates reference interactions at fixed incident photon energies: the energy of the photon immediately before the interaction being sampled. The default grids are:

| Dataset split | Incident photon energy values (MeV) | Number of values |
| --- | --- | ---: |
| Training | 64 geometrically spaced values from 0.001 to 10, plus 1.023, 1.030, 1.050, 1.075 and 1.100 near the pair-production threshold | 69 |
| Validation | Geometric midpoints between neighboring values of the original 64-point grid | 63 |
| Test | 0.010, 0.020, 0.050, 0.100, 0.5, 1, 2, 5 and 10 | 9 |

The 64 geometrically spaced training values, rounded here to six decimal places, are:

| Positions | Incident energies (MeV) |
| --- | --- |
| 1–16 | 0.001000, 0.001157, 0.001340, 0.001551, 0.001795, 0.002077, 0.002404, 0.002783, 0.003221, 0.003728, 0.004314, 0.004994, 0.005780, 0.006690, 0.007743, 0.008962 |
| 17–32 | 0.010372, 0.012005, 0.013895, 0.016082, 0.018614, 0.021544, 0.024936, 0.028861, 0.033405, 0.038664, 0.044750, 0.051795, 0.059948, 0.069386, 0.080309, 0.092951 |
| 33–48 | 0.107584, 0.124520, 0.144122, 0.166810, 0.193070, 0.223463, 0.258642, 0.299358, 0.346483, 0.401028, 0.464159, 0.537228, 0.621800, 0.719686, 0.832981, 0.964111 |
| 49–64 | 1.115884, 1.291550, 1.494869, 1.730196, 2.002568, 2.317818, 2.682696, 3.105013, 3.593814, 4.159562, 4.814372, 5.572265, 6.449467, 7.464760, 8.639884, 10.000000 |

The heads receive photon energy as a continuous logarithmically scaled input within 0.001–10 MeV. Separately, bin boundaries for continuous interaction outcomes, such as scattering angles and energy fractions, using training samples are generated. The validation samples check that representation before training. Test samples are generated separately, with 10 MeV as a training-grid endpoint and 0.100 MeV as a validation midpoint. The test grid is therefore not comprised exclusively of unseen energies.

At each incident energy the generator prepares nine reference sampling sets: one for interaction choice, one for shell choice, one for Rayleigh and one for Compton outcomes, and five photoelectric sets, one per shell. Above the pair threshold it adds one pair-production set. 

Each complete set is a batch of Beam Spinner draws with fixed energy, stochastic quantity and, where applicable, shell. 

The full grids therefore have 69 × 9 + 21 = 642 training sets, 63 × 9 + 16 = 583 validation sets, and 9 × 9 + 3 = 84 test sets. The reduced `--smoke` grids have different totals.

Default samples per set are:

| Setting | Default | Count applies separately to |
| --- | ---: | --- |
| `--categorical-events` | 32,768 | Process selection and shell selection at each photon energy |
| `--continuous-events` | 8,192 | Rayleigh and Compton at each photon energy; photoelectric sampling at each photon energy **and shell** |
| `--pair-events` | 16,384 | Pair production at each photon energy above threshold |

`--smoke` reduces the grids and counts for execution checks. Low statistics can leave representation checks unevaluable, therefore a smoke dataset is not sufficient evidence of physical accuracy.

Generation writes `schema_v4_data.npz`, its manifest and `schema_v4_generator_spec.json`. The NPZ contains samples, category counts, bin edges and metadata. The manifest records the actual generated data; the generator specification describes canonical rules and defaults. Reuse the NPZ across training sessions.

The 13 disjoint heads train one at a time with Adam and cross-entropy. For the default settings, the grouped-distribution heads have up to 400 epochs, while the four pair electron/positron direction heads trained from individual events have up to 20 epochs. After each epoch, the program evaluates cross-entropy on the separately generated validation samples, stops early if it ceases to improve, and restores the weights with the lowest validation cross-entropy, to avoid overfitting. 

All-head training saves individual `v040_head_*.pt` checkpoints, a training manifest and the combined `v040_policy.pt`. Resume a matching dataset/run pair to restore completed heads and train those remaining:

```bash
python -m beamweaver train runs/data/schema_v4_data.npz --resume runs/training
```

Retrain one selected head while retaining the other heads from that policy:

```bash
python -m beamweaver train runs/data/schema_v4_data.npz --resume runs/training --factor comp_u
```

`--factor` selects a head; examples include `process`, `comp_u` and `pair_frac`. Training a single head in a new run leaves the other heads untrained. Always resume with the matching dataset and binning; checkpoint identity checks are not yet comprehensive.

## Simulate, compare and validate

```bash
python -m beamweaver run runs/training/v040_policy.pt --data-dir . --energy 1.0 --histories 500 --output runs/evaluation
python -m beamweaver compare runs/training/v040_policy.pt --data-dir . --output runs/comparison
python -m beamweaver report runs/comparison
```

An audited run saves `dose.npy`, `summary.json`, execution records and, when available, `shower3d.png` showing photon tracks from up to 40 primary histories.

Comparison runs MC1 and MC2 with independent random streams, then Beam Weaver. Defaults are 0.05, 0.5, 1 and 5 MeV, with 2,000 primary photon histories per selected energy per method (MC1, MC2 and Beam Weaver). For four energies this means 4 × 3 × 2,000 = 24,000 primary histories in total. It saves per-simulation energy-deposition arrays and `comparison.json`, including interaction fractions, elapsed time and throughput. `report` creates the comparison figures from those saved outputs.

```bash
python -m beamweaver validate runs/training/v040_policy.pt --data-dir .
python -m beamweaver validate --reference --data-dir .
```

The first command compares learned distributions with reference samples; the second checks reference angular samplers against their analytic targets. Use `python -m beamweaver <command> --help` for all available controls.


## Citation, history and license

(https://doi.org/10.5281/zenodo.18994134)**. Cite the specific archived version used and record its Git commit. Citation metadata for the current code is in [CITATION.cff](CITATION.cff); GitHub downloads and release notes are under [Releases](https://github.com/PedroTelesFCUP/Beam_weaver/releases). 

See [architecture](docs/architecture.md) for the module layout and [contributing](contributing.md) for development guidance.

Pedro Teles — Department of Physics and Astronomy, Faculty of Sciences, University of Porto, Portugal.

The project code is licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for notices and [LICENSE-THIRD-PARTY.md](LICENSE-THIRD-PARTY.md) for the scope of third-party licensing. External data and dependencies retain their respective terms.

## References

[1] Badiali, C., Bilbao, P. J., Cruz, F., and Silva, L. O. (2022). “Machine-learning-based models in particle-in-cell codes for advanced physics extensions.” Journal of Plasma Physics, 88(6), 895880602. [doi: 10.1017/S0022377822001180](https://doi.org/10.1017/S0022377822001180). 

[2] Farmer, J. A., Murray, A., Krotz, J., and McClarren, R. G. (2025). “Generative Monte Carlo Sampling for Constant-Cost Particle Transport.” arXiv preprint, 2512.13965. doi: [10.48550/arXiv.2512.13965](https://doi.org/10.48550/arXiv.2512.13965). 

[3] J. S. Bridle, “Probabilistic Interpretation of Feedforward Classification Network Outputs,” *Neurocomputing* (1990). [doi:10.1007/978-3-642-76153-9_28](https://doi.org/10.1007/978-3-642-76153-9_28).

[4] F. Salvat, *PENELOPE-2018: A Code System for Monte Carlo Simulation of Electron and Photon Transport*, OECD/NEA (2019). [doi:10.1787/32da5043-en](https://doi.org/10.1787/32da5043-en).

[5] D. E. Cullen, J. H. Hubbell and L. Kissel, *EPDL97: The Evaluated Photon Data Library, ’97 Version* (1997). [doi:10.2172/295438](https://doi.org/10.2172/295438). 

[6] J. H. Hubbell et al., “Atomic form factors, incoherent scattering functions, and photon scattering cross sections,” *Journal of Physical and Chemical Reference Data* **4**, 471–538 (1975). [doi:10.1063/1.555523](https://doi.org/10.1063/1.555523).

[7] Berger, M.J., Coursey, J.S., Zucker, M.A., and Chang, J. (2005). ESTAR, PSTAR, and ASTAR: Computer Programs for Calculating Stopping-Power and Range Tables for Electrons, Protons, and Helium Ions (version 1.2.3). National Institute of Standards and Technology, Gaithersburg, MD. Available online at NIST STAR Database.

Beam Spinner’s transport and sampling scheme draws substantially on the published PENELOPE algorithms; the author gratefully acknowledges Francesc Salvat, José M. Fernández-Varea, Josep Sempau and the wider PENELOPE development team.

## AI disclosure

ChatGPT (OpenAI) and Claude (Anthropic) assisted with coding and the preparation of diagrams, figures and documentation. The author reviewed the physics, implementation, analyses and interpretations and assumes full responsibility for the final work.
