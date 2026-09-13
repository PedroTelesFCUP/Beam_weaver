# Structural cleanup record

**Historical structural-cleanup record (11 September 2026).** The subsequent constants consolidation, rest-energy correction, lepton-call repair and completed CPU neural checks are documented in [CONSTANTS_CHANGE_RECORD.md](CONSTANTS_CHANGE_RECORD.md). The verification table below records the earlier pass.

## Source and scope

Source: the uploaded `Beam_weaver_0_4_0 (5)(2).py`, identical to the other full 0.4.0 upload.

- Original: **8,823 lines** in one executable module.
- Current package: **4,480 lines** across 15 functional modules plus `__init__` and `__main__`; the small launcher, tests and documentation are additional.
- 51 original top-level function/class definitions were removed; 83 were retained or adapted across the modules.
- Reduction in application source: **49.2%**. This is a source-size measure, not a performance result.
- Source SHA-256: `e9ac773e89c3ffca8c1de61408c6e53df906abae72df0f1b3781ed6f39408647`.

The standalone `v040_module_final.py`, generator specification and change record were supporting context. The complete uploaded program was the implementation authority. The original document's claims about previous successful runs are not treated as tests executed during this cleanup.

This pass implements the requested structural cleanup and naming improvements. It does not harmonize numerical constants or silently repair the inherited learning/checkpoint defects listed below.

## What was removed

- The old `TransportKernel`, its feature extractor, `PhysicsTransportDistribution`, checkpoint loader, trainers and baseline comparison arm.
- Schema-v3 dataset generation, active-dataset pointers, old filename fallbacks and legacy startup checks.
- Old reports, dashboard and compatibility adapters that had no place in the current workflow.
- Obsolete environment training state, energy-bin truth caches, repeated Compton caches, observation bounds, full observation vectors and the unused vector cross-section helpers that served those vectors.
- Unused `_kn_dcs`, dipole sampler, bremsstrahlung helper and its unreachable tail, condensed-history implementation, electron-only compatibility wrapper and reward dispatcher.
- Reward-score outputs and unused arguments in the old acceptance wrappers. Their live angular reference arrays remain explicit validation functions.
- The unused lepton `n_steps` argument and dead bin-truncation options. No constant value or transport integration step was changed by these removals.
- Import aliases carrying development-version suffixes, duplicate process/shell orderings, startup banners describing an old model and historical review chatter.
- Dataset sanity reports now mark statistically skipped checks with null pass/floor values; skipped conditions are no longer printed as successful measurements. This changes reporting only, not sampling or the sanity threshold.
- Diagnostics with no writer: the permanent-zero batch-transfer timing and minimum-fill audit field. The batched runner still reports its measured fill statistics.

The exact top-level source inventory, including original locations and replacement modules, is [SOURCE_INVENTORY.json](SOURCE_INVENTORY.json). A retained class was also inspected for obsolete methods and attributes; this inventory is not a statement that every line of each retained definition is unchanged.

## Names and responsibilities

| Previous name | Current API and purpose |
| --- | --- |
| `ComptonSampler.sample()` | `ComptonEnergyTransferSampler.sample_energy_transfer()`: sample the transferred energy and outgoing photon energy ratio. |
| `sample_compton()` | `sample_compton_event()`: construct the full photon/recoil-electron collision. |
| `sample_rayleigh`, `sample_photoelectric`, `sample_pair` | `sample_rayleigh_event`, `sample_photoelectric_event`, `sample_pair_event`: complete collision events. |
| `sample_photoelectric_forced` | `sample_photoelectric_event_for_shell`: explicit shell-conditioned event. |
| `photon_interact` | `sample_photon_interaction`: choose a reference process and construct its event. |
| `PenelopeLikeWaterData` | `WaterPhotonData`: photon material tables and interpolation. |
| `pick_shell`, `pick_photo_shell` | `sample_shell`, `sample_photo_shell_index`: distinguish the shell result from its integer index. |
| `PhysicsEvent` | `PhotonCollisionEvent`. |
| `run_mc_shower` | `run_reference_transport`. |
| `run_physics_head_shower`, batched variant | `run_learned_transport`, `run_learned_transport_batched`. |
| `LearnableLogits` | `UnconditionalCategoricalHead`: a categorical distribution with learned logits and no conditioning input. |
| `bw4_generate_schema_v4`, `bw4_train_complete`, `bw4_new_policy` | `generate_dataset`, `train_policy`, `create_policy`. |

The Compton energy sampler and event constructor now sit together in `physics.py`. No old-name aliases are retained in production. Test helpers may map original names to current names solely to compare the supplied source with the cleanup.

Current schema markers, model state-dictionary keys, serialized head names, coordinate labels and deterministic seed strings remain intact. They define the meaning of existing current data and weights. For example, the stored `ray_q2` label still identifies the **logarithm of reduced squared momentum transfer**, not a newly changed physical coordinate.

## Behaviour deliberately preserved

Photon reference sampling, head parameterization and deterministic initialization, transformed-coordinate formulas, grouped/per-event targets, likelihood arithmetic, boundary handling, bank ordering, lepton energy accounting and current constants were retained. Scalar and batched transport still differ in random-number scheduling; they are not claimed to produce the same individual histories.

The learned runner passes only energy into the policy. The small encoding functions preserve the original scalar and batch float32 log-energy rounding and clipping. Physical position, direction, free flight and geometry remain in the runner. This removes unnecessary cross-section/observation construction without changing what the deployed head previously received.

The reference guard was adapted to module boundaries and now covers reference functions, package aliases and class-level samplers. It restores aliases after exceptions, including aliases imported during the guarded interval. Concurrent or nested guards are rejected: it is a process-wide diagnostic, not a thread-local isolation mechanism. Shared lepton transport and tabulated attenuation lookups remain allowed.

Execution reports distinguish caller-declared configuration from instrumented counters. A provenance certificate is not evidence of agreement with an external transport code.

## Workflow changes

- Opening the menu, training on a dataset and regenerating figures do not load water tables or run a full physics preflight.
- Reference sampler validation is explicit: `python -m beamweaver validate --reference --data-dir <tables>`.
- Full policy training saves the complete checkpoint automatically. Run/compare/learned-validation commands select and load it directly on the requested device.
- Fresh training is the default; resume must explicitly name a run directory. This reduces accidental reuse but does not replace missing checkpoint identity checks.
- Output files go to selected run directories. Physics CSV readers are contained in a CLI data-directory context which restores the caller's working directory on failure.
- Comparison uses MC1, MC2 and BeamWeaver. The second MC realization is retained for sampling variability. Each saved result records the seed and raw deposited-energy array.
- Batched timings use `t_energy_encoding` and `t_event_generation`; the latter includes the policy's internal tensor transfer. A separate host-to-device timer is no longer claimed.

## Verification

Checks executed with artificial fixtures and the current local Python environment:

| Check | Evidence |
| --- | --- |
| Source parsing and internal references | All package modules compile; imported package symbols resolve statically, with no unresolved global references. |
| Coordinates and current grids | Maximum coordinate round-trip error `7.638334409421077e-14`; current train/validation/test grid sizes 69/63/9 and grid hash preserved. |
| Reference photon samplers | 20 fixed-seed scenarios, five events each, match original outputs and subsequent random draws exactly. |
| Shared lepton transport | 12 electron/positron scenarios match the original exactly, including deposits, emitted photons, boundary escape, annihilation and RNG state. |
| Complete reference showers | 18 histories at 0.05/2/5 MeV; 707 interactions, 448 banked photons and eight pair events. Full output structures and RNG state match the original exactly. Maximum history closure error `1.4210854715202004e-14` MeV. |
| Angular validation extraction | 49 reference angular arrays exactly match the previous wrappers' distribution outputs. |
| Reduced energy interface | Scalar and batch encodings each checked over 20,006 energies, including clipping and threshold cases; exact equality with the old observation-derived energies. |
| Current neural code structure | AST comparison confirmed retained initialization and scalar/batched sampling cores, checkpoint methods and likelihood arithmetic, allowing the documented naming and interface changes. |
| Dataset workflow | Actual command-line generation completed with 217 artificial condition groups; saved splits, bin-edge hashes and dataset hash checked. The low-statistics representation checks were explicitly skipped. |
| Packaged tests | 10 tests: six passed, four neural tests skipped because PyTorch is absent. |
| Sampler guard | Module aliases, class methods, exception restoration, nested-entry rejection and aliases imported during the guard are checked. |

**Not executed:** real neural initialization/training/inference, CUDA tests, current checkpoint reload with a real model, or production-table shower comparisons. PyTorch, production water tables, datasets and checkpoints are absent here. Supplied neural integration tests explicitly skip when PyTorch is unavailable; a skip is not a pass. The user should run them in the training environment before replacing a running experiment.

The synthetic fixtures test extraction and conservation. They cannot establish physical accuracy in water, statistical distribution agreement against PENELOPE, performance improvement, or preservation of a production PDD.

## Known issues intentionally left for the next numerical/integrity pass

1. **Masked cross-entropy:** `_masked_ce` still turns all non-finite log probabilities into zero. This can hide invalid targets, NaNs or all-masked rows. The correct fix needs explicit support/finiteness checks and careful handling of zero-target terms.
2. **Factor checkpoint identity:** `restore_factor` does not verify dataset identity, coordinates or bin-edge values. Use only the matching dataset/run directory until this is fixed.
3. **Complete checkpoint edges:** duplicate edge representations and stored hashes are not cross-validated by the inherited loader. Equal tensor dimensions do not establish equal decoder meaning.
4. **Training failure state:** the factor trainer can leave changed `requires_grad` flags after an exception. This needs input validation before mutation and restoration in `finally`.
5. **Histogram support:** inherited target construction can silently discard out-of-range samples; a separate input-validation change should reject or explicitly account for them.
6. **Constants:** `mec2 = 0.51099895069` MeV and `MEC2_MEV = 0.5109989` MeV are still separate. Their 0.05069 eV difference, repeated hc values and duplicated scalar interpolators should be consolidated in a controlled numerical change.

These are inspection findings confirmed in the supplied prior audit; Torch-based reproductions of them were not rerun here. This package does not hide them behind a claim of publication readiness.
