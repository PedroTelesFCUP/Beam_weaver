# Review patches

The final package is already updated. These patches are separate review records for the two requested changes and the prerequisite lepton-call repair.

Apply in numeric order only to the modular package from before this constants pass, including its restored startup banner:

```bash
patch -p1 < /path/to/00-repair-lepton-call-sites.patch
patch -p1 < /path/to/01-consolidate-constants.patch
patch -p1 < /path/to/02-harmonize-rest-energy.patch
```

Paths are relative to the package root. Each patch covers application/test Python source and `pyproject.toml`. Documentation and verification artifacts are provided separately in the final archive. `patch_manifest.json` records before/after SHA-256 values for each changed file.

See `../CONSTANTS_CHANGE_RECORD.md` for scope, results and remaining limits.
