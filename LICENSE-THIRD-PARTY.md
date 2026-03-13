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


### Publications
Manuscripts, figures, and supplementary materials may carry separate copyright or publisher terms.

## Recommended practice
For public release:

1. keep original research code under Apache-2.0
2. keep third-party data and documents clearly separated
3. add attribution and source links for any redistributed tables
4. avoid bundling copyrighted manuals unless redistribution is explicitly allowed
