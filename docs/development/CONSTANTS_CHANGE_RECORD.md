# Constants consolidation and rest-energy correction

Completed 12 September 2026. This package contains both requested changes. The review patches separate the behavior-preserving consolidation from the intentional rest-energy correction. A two-line prerequisite repair is recorded separately because full neural shower execution exposed it during verification.

## Review order

| Patch | Change | Numerical intent |
| --- | --- | --- |
| `review/00-repair-lepton-call-sites.patch` | Remove the obsolete `n_steps=5` argument from the two learned-runner calls to `transport_lepton_csda`. The function no longer accepts it, and the original parameter was unused. | Restores execution; no physics formula or stepping algorithm changes. |
| `review/01-consolidate-constants.patch` | Consolidate genuine duplicate values, conversions, indices, dimensions and matching configuration defaults. | Preserve values, arithmetic, seeded draws, categories, caller overrides and outputs. |
| `review/02-harmonize-rest-energy.patch` | Retire `MEC2_MEV`, use `mec2` everywhere, derive pair support and record numerical provenance. Add targeted boundary/accounting tests. | Intentional numerical correction in learned reconstruction and pair support. |

The archive already contains the final code. These patches are for review or for applying, in order, to the modular package as it stood before this constants pass (including the restored startup banner). Do not apply them again to the supplied final code. Their scope is Python application/test source and package build configuration; the accompanying documentation and verification reports describe the completed result.

## Change 1: one definition for each shared quantity

- `hc` has one numerical default, with its MeV·angstrom form derived from its keV·angstrom form. `WaterPhotonData.HC_KEV_A` takes that default while retaining existing instance overrides.
- Binding energies are explicitly named in eV; their MeV mapping is derived once. Named unit factors replace actual energy-unit conversions.
- Shell/process mappings and model dimensions derive from the existing ordered lists. Every event still receives a fresh shell vector. No category, head name, state-dictionary layout or edge interpretation changes.
- Forward direction rotation and inverse azimuth extraction share the same `0.9999` branch threshold.
- Source width/half-width, phantom slab spacing, policy-domain defaults and matching transport defaults use shared or derived definitions. Source draws, arithmetic, energy clipping and scalar/batch float32 encoding retain their previous behavior.
- Matching attenuation, history-closure and learned reconstruction guards have descriptive names. Different units and scientific roles retain independent controls, including transport cutoffs, lepton step lengths, model bounds and training tolerances.
- Matching command defaults, table filenames and configuration descriptions use their authoritative definitions. Build metadata and the displayed version derive from `BW_VERSION`. The startup banner remains unchanged in appearance, with Pedro Teles as the displayed author.
- Fixed random-seed namespaces remain unchanged. Equal factor-specific bin counts/sample counts remain separate controls. The table convention `x = q/2` remains intact.

The comparison baseline for this stage contains only the separately documented two-line lepton-call repair. The untouched pre-change package is retained in the development workspace; no compatibility wrapper was used for the final comparison.

## Change 2: canonical electron rest energy

```python
mec2 = 0.51099895069  # MeV
PAIR_THRESHOLD_MEV = 2.0 * mec2
```

The retired value was `0.5109989` MeV. The rest-energy correction is `0.05069 eV`, and the pair threshold changes from `1.0219978` to `1.02199790138` MeV (`0.10138 eV`). Reference photon sampling and the coordinate transforms already used the chosen value. Learned Compton reconstruction and pair kinetic/rest-energy accounting now agree with it.

All default training, validation and test energies retain their pair-eligibility classification. Arbitrary energies in the narrow interval between the thresholds can change classification, as intended.

The existing comparison operators are preserved: policy/training masks exclude the threshold itself; the reference pair sampler retains its existing `<` check and can return a degenerate zero-kinetic pair at the boundary. Scalar/batch policy tests check both direct physical-energy input and the existing transport encoding routes. Equal float32 model features do not determine the physical support mask.

New datasets and generator specifications record `physics_constants`. New complete checkpoints and run/comparison/execution reports record `runtime_physics_constants`, describing the current reconstruction code. Factor checkpoints record `training_physics_constants` when that factor is actually trained. This does not relabel the training history of loaded weights. Old files without these fields remain loadable and are not rewritten by loading them.

Existing checkpoint shapes, category order and edges are unchanged. No automatic dataset regeneration or retraining is performed. Learned shower outputs are not claimed to remain bitwise identical after this intentional correction.

## Verification completed

CPU execution used Python 3.12, NumPy 2.3.5 and PyTorch 2.14.0+cpu. Artificial tables and small artificial training targets exercise execution, reproducibility and conservation; they are not a water-physics validation dataset.

| Check | Result |
| --- | --- |
| Stage 1 dataset comparison | 629 saved arrays, including bin edges, match exactly. |
| Stage 1 neural comparison | All 13 heads initialize, train and restore identically; losses, weights, 128 scalar draws, 128 batched draws and RNG states match. |
| Stage 1 complete transport comparison | 18 reference histories (707 interactions, 8 pairs) and 36 learned scalar/batched histories (1,350 interactions, 6 pairs) match exactly, including photon banks, ledgers and RNG states. |
| Original reference engine comparison | Same 18 reference histories, complete outputs and RNG states match the supplied original source exactly. |
| CLI and banner | Identical banner, 7 help outputs, 9 parsed command cases and 6 menu scenarios. |
| Direction/source/encoding contracts | Both polar branches, independent shell vectors, caller settings, exact source draws, and 60,018 energies in each encoding mode across three domains pass. |
| Stage 2 pair support | Direct and encoded scalar/batch routes pass around float64/float32 boundaries and in the interval between the old and chosen thresholds. All five pair heads are checked. |
| Stage 2 energy accounting | Actual scalar and batched runners pass Compton recoil momentum/energy, pair kinetic splitting, CSDA annihilation, bank accounting and below-threshold rejection checks. |
| Provenance and training masks | Canonical units/values, fresh metadata snapshots, six threshold-adjacent training conditions, trained-factor records and metadata-free checkpoint loading pass. |
| Full test suite | 29 tests passed, none skipped when CPU Torch and the supplied original source were available. |
| Package integrity | Internal imports resolve; package metadata builds into a wheel with the canonical version. Review patches are checked by applying them in order and comparing resulting files with the final source. |

Machine-readable evidence and test transcripts are in `verification/`. The old golden reference events were retained. The test comparison explicitly removes only the retired learned-rest-energy metadata field; it does not regenerate reference event or lepton-ledger expectations.

CUDA, production water tables, a production checkpoint/PDD, and external Monte Carlo comparisons were not available for this pass. Those remain the limits of the verification. The earlier masked-cross-entropy, checkpoint-identity and training-error-state findings remain separate work; they were not changed here.

## Using the result

Replace the complete package with this archive so imports and shared definitions stay consistent. Keep your physics tables, datasets and checkpoints in their existing separate directories. Launch with `python -m beamweaver` or the included `Beam_weaver_0_4_0.py` launcher. No review patch needs to be applied to the final package.
