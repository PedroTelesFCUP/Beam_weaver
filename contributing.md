# Contributing to Beam Weaver

Contributions can improve collision samplers, learned distributions, numerical
validation, performance, documentation and reproducibility.

For a bug report, include the command or menu choices, Python and dependency
versions, photon energy, seed, table/dataset/checkpoint provenance, and the
smallest example that reproduces the problem. Include any traceback.

Develop changes on a branch and open a pull request explaining the problem,
the resulting behavior and how it was checked. Keep structural changes
separate from intentional physics or numerical changes. Preserve category
ordering, random-stream behavior and caller overrides unless changing them
is an explicit part of the proposal.

Install with `python -m pip install -e '.[learn]'` and run
`python -m unittest discover -s tests -v`. See [tests/README.md](tests/README.md)
for coverage and the optional original-source comparison. Artificial fixtures
check execution and regression; physics changes also need distribution,
threshold and energy-accounting evidence appropriate to the change.

The current application uses supervised cross-entropy training of thirteen
disjoint heads. See [architecture](docs/architecture.md) for the module layout
and [development records](docs/development) for the cleanup and constants checks.
