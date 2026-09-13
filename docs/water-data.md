# Water tables

The six root CSVs are preserved byte-for-byte from commit
`67a5af859c95e5f09cafe460b53d806a181dcc1c`. The repository refresh changes their
packaging and documentation, not their contents.

| Tables | Provenance recorded in the earlier repository |
| --- | --- |
| Rayleigh, Compton and pair/triplet cross sections | The earlier manuscript describes PENELOPE-derived tables. |
| Photoelectric cross sections and shell data | The manuscript describes EPDL data; the shell file contains hydrogen K and oxygen K, L1, L2 and L3 contributions. |
| `water_sq.csv` | The earlier README cites Hubbell et al. (1975), using `2*S_H + S_O`. |
| `water_fq.csv` | The earlier README describes Cromer–Mann-style interpolation. A more specific extraction provenance is not established by the documentation. |
| `ElectronStoppingPower.csv` | The exact source and conversion procedure are not established by the documentation. |

This records existing attribution; it is not an independent verification of
the tabulations. The original manuscript and bibliography remain in the
[preserved SAC snapshot](https://github.com/PedroTelesFCUP/Beam_weaver/tree/archive/sac-before-0.4.0/paper).

Photon and shell tables span 0.001–1000 MeV. Stopping powers span 0.01–1000 MeV;
the current interpolation uses the first table value below 10 keV. The learned
policy domain remains 0.001–10 MeV. In both scattering-function tables, the
column called `q` is the table coordinate `x`; the samplers evaluate it at
`x = q / 2` for full momentum transfer `q`.

The pair cross section is tabulated as zero at 1.022 MeV and first positive at
1.03 MeV. Current interpolation returns zero between a zero endpoint and the
next positive endpoint, so a positive kinematic pair allowance does not imply
positive process-selection probability throughout that interval. This
behavior predates the repository refresh.

Current loaders accept all six tables. Column, finite-value, ordering and basic
interpolation checks establish execution compatibility, not physical
validation. External table terms remain separate from the project code's
Apache license; see [LICENSE-THIRD-PARTY.md](../LICENSE-THIRD-PARTY.md).
