# Material tables

Any user-defined table can be used as long as with the proper format. In the current implementation, only water was used, but other materials can and will be used in the future (and can be manually created by users).

Future work will produce these tables automatically or eventually providing tables for different materials.

# Water Values

Water values were taken from different sources, in order to improve the results. Earlier versions used XCOM values for cross-sections and analytical approximation values for the form factors. The current implementation uses values taken directly from the EPDL database for Compton, Photo, and Pair Production. Rayleigh cross-section values were taken from the Penelope 'water.mat' information file. Pair production is the sum between duplet and triplet values. Stopping Power values were taken from the NIST website (ESTAR database).

| Tables | Provenance recorded in the earlier repository |
| --- | --- |
| Compton and pair/triplet cross sections | EPDL data |
| Photoelectric cross sections and shell data | EPDL data|
| Rayleigh | Penelope water.mat data |
| `water_sq.csv` | Hubbell et al. (1975), using `2*S_H + S_O`. |
| `water_fq.csv` | Hubbell et al. (1975), using `2*F_H + F_O`. |
| `ElectronStoppingPower.csv` | ESTAR (NIST data) |


Photon and shell tables span 0.001–1000 MeV. Stopping powers span 0.01–1000 MeV;
the current interpolation uses the first table value below 10 keV. The learned
policy domain remains 0.001–10 MeV. In both scattering-function tables, the
column called `q` is the table coordinate `x`; the samplers evaluate it at
`x = q / 2` for full momentum transfer `q`.

The pair cross section is tabulated as zero at 1.022 MeV and first positive at
1.03 MeV. Current interpolation returns zero between a zero endpoint and the
next positive endpoint, so a positive kinematic pair allowance does not imply
positive process-selection probability throughout that interval. 

External table terms remain separate from the project code's
Apache license; see [LICENSE-THIRD-PARTY.md](../LICENSE-THIRD-PARTY.md).
