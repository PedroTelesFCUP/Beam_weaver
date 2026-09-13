# Changes in 0.4.0

The current application learns thirteen disjoint stochastic heads from Monte
Carlo collision samples using cross-entropy. The public workflow is generate
data, train, simulate, compare and validate.

- Replace the SAC-era monolith with the modular `beamweaver` package.
- Provide six menu choices and matching commands, including selected-head
  retraining and explicit sample counts per photon energy and shell.
- Share numerical constants and derive the pair threshold from
  `mec2 = 0.51099895069` MeV. The intentional rest-energy correction is recorded
  separately from the behavior-preserving consolidation.
- Include audited transport, independent reference comparisons, saved
  provenance and regression checks.
- Retain the six existing water tables and license notices.
- Update installation, citation information and documentation for the method.
- Restore the all-versions Zenodo DOI and prepare a tested 0.4.0 release.
- Document the method and add v0.4.0 results at five primary energies,
  original 5 MeV figures and reproducible plot data.

The complete preceding repository, including the SAC implementation and its
manuscript, is preserved at
[`archive/sac-before-0.4.0`](https://github.com/PedroTelesFCUP/Beam_weaver/tree/archive/sac-before-0.4.0).
Existing `v0.1.0` and `v0.1.0.1` tags remain available. The historical
[Zenodo record](https://doi.org/10.5281/zenodo.18994135) is retained as a reference
to the earlier software; it is not presented as an archive of this 0.4.0 code.

Version **0.4.0** is archived at [doi:10.5281/zenodo.22739031](https://doi.org/10.5281/zenodo.22739031).
The [all-versions DOI](https://doi.org/10.5281/zenodo.18994134) follows the project’s archive history.

Detailed cleanup and constants evidence is in [docs/development](docs/development).
