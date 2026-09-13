Run from the project directory:

```bash
python -m unittest discover -s tests -v
```

The tests create tiny artificial material tables in temporary directories.
These are execution and regression checks, not physical validation of water
transport. No production tables, training dataset, or checkpoint is required.

- The reference golden test covers twenty seeded photon sampler scenarios,
  twelve electron/positron transport cases, coordinate round trips, and
  zero-support interpolation. Golden floating-point outputs use relative
  tolerance `1e-12` and absolute tolerance `1e-14`; categories, structures,
  and subsequent RNG draws must match exactly.
- Four guard tests check reference entry points, module aliases, shell
  selectors, exception cleanup, nested-entry rejection, and restoration of
  aliases imported while the guard is active.
- Neural tests train all thirteen heads for one epoch on artificial
  categorical targets, check readiness and pair support, and check complete
  checkpoint round trips. If Torch is unavailable these tests are explicitly
  skipped. Passing the non-neural tests does not establish neural execution.

The optional original-source neural comparison requires the previous source
file supplied by the user:

```bash
BEAMWEAVER_ORIGINAL=/absolute/path/to/original.py python -m unittest discover -s tests -v
```

It compares initialization, one grouped-head and one per-event training
update, seeded scalar and batched collision generation, and RNG state.

An exact full-shower comparison is also available when that source is supplied:

```bash
python tests/compare_original_shower.py /absolute/path/to/original.py --package-dir .
```

This uses artificial tables and six histories at each of 0.05, 2, and 5 MeV.
It compares all output records, dose arrays, photon-bank histories, and RNG
state exactly. Both runners use the current phantom, so it isolates transport
and its reference sampling calls. The old source is not included in this
package and is never needed by the production application.

`test_dataset_generation.py` exercises the full saved-dataset pipeline using artificial tables, including edge/data hashes and explicit skipped-check metadata.

The constants pass adds tests for source/frame/encoding contracts, canonical rest-energy reconstruction and pair masks in both runners, and numerical provenance in datasets/checkpoints. The complete suite passed all 29 tests with CPU Torch and the supplied original source. Existing checkpoint files are not modified by the loading tests. The optional original-source check is still explicitly skipped if that source is not supplied.
