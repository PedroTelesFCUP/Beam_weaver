# Third-party licenses and content boundaries

The **Apache-2.0** license in this repository applies to the original code authored for this project.

It does **not** automatically relicense:

- third-party Python packages used by this project
- external data tables
- external Monte Carlo reference code
- external PDFs, manuals, or publications
- any separately distributed pretrained weights or datasets whose origin carries distinct terms

## Important boundaries

### Dependencies
Typical runtime dependencies such as PyTorch, Gymnasium, NumPy, pandas, Matplotlib, IPython, and Stable-Baselines3 remain under their own licenses.

### Data files
Files such as:

- `NIST_WaterCrossSections.csv`
- `OxygenPhotoShells.csv`
- `ElectronStoppingPower.csv`
- `Final_cross_sections.csv`
- `Rayleigh_cross_sections.csv`

should only be redistributed if you are sure you have the right to do so, and with their original attribution / terms intact.

### PENELOPE-related materials
If you compare against PENELOPE or consult the PENELOPE manual, do not imply that this repository contains official PENELOPE source code unless it actually does and you have the right to redistribute it.

### Publications
Manuscripts, figures, and supplementary materials may carry separate copyright or publisher terms.

## Recommended practice
For public release:

1. keep original research code under Apache-2.0
2. keep third-party data and documents clearly separated
3. add attribution and source links for any redistributed tables
4. avoid bundling copyrighted manuals unless redistribution is explicitly allowed
