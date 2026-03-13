# Contributing to Beam Weaver

Thank you for your interest in contributing to **Beam Weaver**.

Beam Weaver was developed over several months as an internal research
project by the authors and is now being shared with the community to
encourage visibility, discussion, external scrutiny, and future
collaboration.

At this stage, Beam Weaver should be understood as a **research
prototype**. The codebase is already functional for its intended
research setting, but it remains a work in progress and some components
are still under refinement.

## Scope of contributions

We welcome contributions that improve Beam Weaver as a research
software project, including:

- bug reports and debugging help
- documentation improvements
- installation and reproducibility improvements
- code cleanup and refactoring
- test scripts and validation workflows
- benchmarking and evaluation utilities
- visualization tools
- physics-model review and consistency checks
- reinforcement-learning improvements that are clearly documented
- discussion of transport modeling assumptions and limitations

Particularly helpful contributions include those that improve clarity,
reproducibility, maintainability, and scientific transparency.

## Before contributing

Please keep in mind the following:

- Beam Weaver is currently released for **research and development**
  purposes.
- It is **not validated for clinical, safety-critical, or production
  use**.
- Contributions should preserve scientific clarity and should not
  overstate the current maturity or validation status of the software.

## How to contribute

### Architecture and engineering contributions are especially welcome

Beam Weaver was developed primarily as a focused research codebase, and
parts of the current implementation remain very monolithic.

One of the most valuable forms of contribution at this stage would be
to help evolve Beam Weaver into a more modular, maintainable, and
release-ready software project. This includes, for example:

- separating core physics, learning, training, and evaluation logic
  into clearer components
- improving code organization and internal APIs
- reducing monolithic script structure
- improving configuration handling
- adding tests and reproducibility checks
- improving packaging, installation, and versioned releases
- supporting cleaner interfaces for future extension and collaboration

Contributors interested in software engineering, architecture, and
research-software sustainability are therefore especially encouraged to
get involved.

The current public release reflects a substantial and functional
research prototype, but part of the long-term goal is to make Beam
Weaver less monolithic and progressively closer to a robust,
well-structured software release suitable for broader research use and
community development.

### Reporting issues

If you find a bug, inconsistency, or unclear behavior, please open a
GitHub Issue and include as much detail as possible:

- what you were trying to do
- what happened
- what you expected to happen
- steps to reproduce the problem
- relevant logs, stack traces, or screenshots
- operating system and Python environment details

### Suggesting improvements

Feature suggestions are welcome. Please open an Issue first for major
changes so that the direction can be discussed before substantial work
is done.

The most important aspect now is the angular deflection training.

### Submitting code changes

If you want to contribute code:

1. Fork the repository.
2. Create a new branch for your change.
3. Make your edits in a focused and clearly documented way.
4. Test the change as far as possible.
5. Open a Pull Request with a clear explanation of what was changed and
   why.

Please keep pull requests relatively small and targeted whenever
possible.

## Coding expectations

When contributing code, please try to:

- write clear and readable Python
- keep functions and classes well documented
- avoid unnecessary complexity
- explain non-obvious design choices
- preserve compatibility with the existing research workflow unless a
  deliberate change is being proposed
- clearly identify any change that may affect numerical behavior,
  transport modeling, or training dynamics

If a contribution changes physics handling, reward structure, model
training, binning logic, or data assumptions, please state that
explicitly in the Pull Request.

## Reproducibility and scientific changes

Beam Weaver is a research codebase, so reproducibility matters.

If your contribution affects:
- transport physics
- interaction probabilities
- angle sampling
- secondary particle handling
- reward calculation
- replay or n-step learning logic
- evaluation outputs

please document the change carefully and, where possible, describe how
the effect can be checked or reproduced.

## Documentation contributions

Documentation improvements are very welcome.

This includes:
- README improvements
- installation instructions
- example workflows
- comments in code
- clarification of input files
- explanation of training and evaluation outputs

## Community standards

Please be respectful, constructive, and specific in discussions.

Good-faith critique is welcome, especially for:
- physics assumptions
- methodological limitations
- reproducibility concerns
- software design trade-offs

## Questions

If you are unsure whether a contribution is appropriate, open an Issue
first and describe the idea briefly.

## A note from the authors

Beam Weaver is being shared with the community after an extended period
of focused independent development. We are releasing it in this form
because we believe the project has a distinctive and promising research
setting, and we hope that making it public will support discussion,
reuse, validation, and eventual collaboration.