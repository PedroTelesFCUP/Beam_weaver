# Beam Weaver

**Learned stochastic kernels for Monte Carlo photon transport.**

Beam Weaver learns the probability distributions governing individual photon collisions from reference Monte Carlo samples. It then samples those learned distributions during transport, generating photon histories and charged-particle secondaries in water. The reference Monte Carlo engine, **Beam Spinner**, supplies the training examples and the conventional transport used for comparison.

The workflow is **generate data → train → simulate and compare**. The current implementation is version **0.4.0**.

## Method

Thirteen heads with disjoint parameters represent interaction selection, photoelectric shell selection, and the energy-sharing and angular variables of Rayleigh scattering, Compton scattering, photoelectric absorption and pair production. Each head learns a categorical distribution; continuous variables are sampled within the selected bin and transformed back to physical quantities.

Training minimizes cross-entropy against reference samples or their empirical category distributions. The heads use photon energy and, where required, the selected shell or sampled lepton energy fraction. Training proceeds head by head, with validation-based stopping and restoration of the best weights.

During a shower, Beam Weaver supplies the collision variables. Free-flight distances still use tabulated attenuation coefficients, and explicit kinematics reconstruct the outgoing particles. Both learned and reference simulations use the same approximate electron/positron transport. The learned energy domain is **0.001–10 MeV**.

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

Each operation asks for its inputs and output directory. Help and the menu can be opened without physics tables or a checkpoint.

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

## Generate and train

Commands support the same workflow as the menu. Use a new or empty output directory; omit `--output` to create a timestamped directory under `runs`.

```bash
python -m beamweaver generate --data-dir . --output runs/data
python -m beamweaver train runs/data/schema_v4_data.npz --output runs/training
```

Generation samples individual collisions at fixed photon energies, producing separate training, validation and test samples. Default sample counts are:

| Setting | Default | Count applies separately to |
| --- | ---: | --- |
| `--categorical-events` | 32,768 | Process selection and shell selection at each photon energy |
| `--continuous-events` | 8,192 | Rayleigh and Compton at each photon energy; photoelectric sampling at each photon energy **and shell** |
| `--pair-events` | 16,384 | Pair production at each photon energy above threshold |

These are separate counts, not a combined event total. `--smoke` reduces the grids and counts for execution checks. Low statistics can leave representation checks unevaluable; a smoke dataset is not sufficient evidence of physical accuracy.

Generation writes `schema_v4_data.npz`, its manifest and `schema_v4_generator_spec.json`. The NPZ contains samples, category counts, bin edges and metadata. The manifest records the actual generated data; the generator specification describes canonical rules and defaults. Reuse the NPZ across training sessions.

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

Comparison runs **MC1 and MC2 with independent random streams**, then Beam Weaver. Defaults are 0.05, 0.5, 1 and 5 MeV, with 2,000 primary histories per simulation at each energy. It saves per-simulation energy-deposition arrays and `comparison.json`, including interaction fractions, elapsed time and throughput. `report` creates the comparison figures from those saved outputs.

```bash
python -m beamweaver validate runs/training/v040_policy.pt --data-dir .
python -m beamweaver validate --reference --data-dir .
```

The first command compares learned distributions with reference samples; the second checks reference angular samplers against their analytic targets. Use `python -m beamweaver <command> --help` for all available controls.

## Interpretation and limitations

The fixed phantom spans x,y = −50…50 cm and z = 0…100 cm, with a uniform 10 × 10 cm² incident photon field. Its 100 depth slabs are 1 cm thick. Arrays named `dose` contain **deposited energy in MeV**, not absorbed dose in Gy. Absolute dosimetry requires mass and source normalization. Report figures normalize each depth profile to its own maximum and compare shape.

The MC1–MC2 difference illustrates sampling variability; it is not a confidence interval. Execution auditing checks covered photon-sampler calls and energy accounting, rather than establishing physical accuracy. Electron/positron transport uses continuous slowing down and simplified bremsstrahlung; positrons use electron stopping powers and annihilate at rest. The combined pair/triplet channel uses pair-event kinematics. These approximations require assessment against a trusted transport code for the intended application.

Run regression checks with `python -m unittest discover -s tests -v`. Neural checks require PyTorch; optional checks against an original source file require that file. Tests report unavailable prerequisites as skips. See `tests/README.md` for coverage.

## Citation, history and license

Citation metadata for the current code is in [CITATION.cff](CITATION.cff). Record the version and commit used for reproducible results. The [changelog](CHANGELOG.md) links the preserved earlier releases and distinguishes their Zenodo record from the current implementation.

See [architecture](docs/architecture.md) for the module layout and [contributing](contributing.md) for development guidance.

Pedro Teles — Department of Physics and Astronomy, Faculty of Sciences, University of Porto, Portugal.

The project code is licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for notices and [LICENSE-THIRD-PARTY.md](LICENSE-THIRD-PARTY.md) for the scope of third-party licensing. External data and dependencies retain their respective terms.
