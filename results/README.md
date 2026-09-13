# Results

[Back to the project README](../README.md)




## Reading the plots

- **Navy:** Beam Spinner MC1. **Blue:** independent MC2 (dashed where curves are used). **Orange:** Beam Weaver.
- **PDD:** each deposited-energy profile is divided by its own maximum. The 100 bins are 1 cm deep. Raw arrays contain deposited energy in MeV, not absorbed dose in Gy.
- **Angles:** probabilities in eighteen 10° bins, accumulated over collisions throughout each recursive shower. The energy label identifies the primary photon, not every individual collision energy. The saved ASCII probabilities are rounded and preserved in the numeric data; each displayed histogram is normalized to sum one. Reported KS statistics come from the saved summaries.
- **Shell selection:** fractions of recorded photoelectron tags in H-K, O-K, O-L1, O-L2 and O-L3; the horizontal axis is logarithmic.
- **Pair sharing:** thirty equal bins of electron kinetic fraction, $f=T_-/(T_-+T_+)$. Adjacent electron and positron secondary records are matched from the same pair event.
- **Interaction fractions:** fractions of recorded events throughout the recursive shower, with a separate enlarged pair-production panel. They are not cross-section ratios at the primary energy alone.

Error bars are omitted. MC1–MC2 differences illustrate finite-sampling variability and are not confidence intervals. The two reference runs and Beam Weaver share photon free-flight sampling and approximate electron/positron transport.

## Recorded execution times

These are wall-clock times in seconds from the saved campaign. Hardware information was not recorded. Beam Weaver took longer than either reference run at every energy in these particular measurements.

| Energy (MeV) | MC1 (s) | MC2 (s) | Beam Weaver (s) |
| ---: | ---: | ---: | ---: |
| 0.1 | 26.84 | 26.22 | 285.13 |
| 1 | 175.09 | 175.86 | 644.25 |
| 2 | 414.80 | 413.81 | 979.50 |
| 5 | 1090.10 | 1097.00 | 1919.30 |
| 10 | 2160.10 | 2153.90 | 3651.90 |

## 0.1 MeV

### Peak-normalized depth dose

![0.1 MeV — Peak-normalized depth dose](figures/0p1MeV/pdd.png)

### Compton polar angle

![0.1 MeV — Compton polar angle](figures/0p1MeV/compton_angle.png)

### Photoelectric polar angle and shell selection

![0.1 MeV — Photoelectric polar angle and shell selection](figures/0p1MeV/photoelectric_angle_shell.png)


### Interaction fractions

![0.1 MeV — Interaction fractions](figures/0p1MeV/interaction_fractions.png)


## 1 MeV

### Peak-normalized depth dose

![1 MeV — Peak-normalized depth dose](figures/1MeV/pdd.png)

### Compton polar angle

![1 MeV — Compton polar angle](figures/1MeV/compton_angle.png)

### Photoelectric polar angle

![1 MeV — Photoelectric polar angle](figures/1MeV/photoelectric_angle_shell.png)


### Interaction fractions

![1 MeV — Interaction fractions](figures/1MeV/interaction_fractions.png)


## 2 MeV

### Peak-normalized depth dose

![2 MeV — Peak-normalized depth dose](figures/2MeV/pdd.png)

### Compton polar angle

![2 MeV — Compton polar angle](figures/2MeV/compton_angle.png)

### Photoelectric polar angle

![2 MeV — Photoelectric polar angle](figures/2MeV/photoelectric_angle_shell.png)


### Interaction fractions

![2 MeV — Interaction fractions](figures/2MeV/interaction_fractions.png)


## 5 MeV

### Peak-normalized depth dose

![5 MeV — Peak-normalized depth dose](figures/5MeV/pdd.png)

[Original vector PDF](figures/5MeV/pdd.pdf)

### Compton polar angle

![5 MeV — Compton polar angle](figures/5MeV/compton_angle.png)

[Original vector PDF](figures/5MeV/compton_angle.pdf)

### Photoelectric polar angle and shell selection

![5 MeV — Photoelectric polar angle and shell selection](figures/5MeV/photoelectric_angle_shell.png)

[Original vector PDF](figures/5MeV/photoelectric_angle_shell.pdf)

### Pair kinetic-energy sharing

![5 MeV — Pair kinetic-energy sharing](figures/5MeV/pair_share.png)

[Original vector PDF](figures/5MeV/pair_share.pdf)

### Interaction fractions

![5 MeV — Interaction fractions](figures/5MeV/interaction_fractions.png)

[Original vector PDF](figures/5MeV/interaction_fractions.pdf)


## 10 MeV

### Peak-normalized depth dose

![10 MeV — Peak-normalized depth dose](figures/10MeV/pdd.png)

### Compton polar angle

![10 MeV — Compton polar angle](figures/10MeV/compton_angle.png)

### Photoelectric polar angle

![10 MeV — Photoelectric polar angle](figures/10MeV/photoelectric_angle_shell.png)


### Interaction fractions

![10 MeV — Interaction fractions](figures/10MeV/interaction_fractions.png)

## Data and reproduction

The compact [campaign data](data/campaign.json) contains the numeric dose arrays, angular probabilities, shell counts, pair-sharing histogram counts, timing and original run identifiers used here. Each source file is identified by its name, energy, byte size and SHA-256 hash. The [original figure manifest](data/original_figures.json) records the exact original PNG/PDF hashes and source archive. 

From the repository root, with NumPy and Matplotlib installed:

```bash
python scripts/plot_results.py
```

The plotter regenerates the figures from these compact numeric inputs and preserves the five original 5 MeV PNGs. See `python scripts/plot_results.py --help` for a separate output directory and for regenerating a comparison set at 5 MeV. No checkpoint or simulation is needed for plotting. Large original pickle archives are not needed or loaded by this plotter.

