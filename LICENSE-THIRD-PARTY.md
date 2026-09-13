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
Runtime dependencies NumPy, pandas and Matplotlib, and the optional learning dependency PyTorch, remain under their own licenses.

### Water tables

The six CSV tables are retained unchanged from the earlier repository. See [docs/water-data.md](docs/water-data.md) for the documented sources and table conventions. The code license does not change the terms of external data.


### Publications
Manuscripts, figures, and supplementary materials may carry separate copyright or publisher terms.

## Recommended practice
For public release:

1. keep original research code under Apache-2.0
2. keep third-party data and documents clearly separated
3. add attribution and source links for any redistributed tables
4. avoid bundling copyrighted manuals unless redistribution is explicitly allowed
