# Reproducibility checklist

This project should be reproducible from a fresh Git clone.

## Paths and imports

- [ ] No machine-specific absolute paths
- [ ] Package imports work without IDE-specific PYTHONPATH settings
- [ ] Commands do not depend on an undocumented working directory
- [ ] Project package is installed through pyproject.toml

## Dependencies

- [ ] Every required dependency is declared
- [ ] A fresh environment can be created from project files
- [ ] `python -m pip check` passes
- [ ] Final dependency reproducibility strategy is chosen:
  - lock file, or
  - constraints file, or
  - pinned requirements

## Data

- [ ] Dataset source is documented
- [ ] Dataset can be reproduced after a fresh clone
- [ ] Required local data files are not assumed to exist
- [ ] Dataset licensing / redistribution rules are respected

## Artifacts

- [ ] Required model artifacts are either reproducibly generated or versioned
- [ ] Large artifacts use Git LFS when necessary
- [ ] `git lfs pull` is documented when LFS is used
- [ ] LFS files are checked to ensure real binaries were downloaded, not pointer files

## Documentation

- [ ] No undocumented manual setup steps
- [ ] README contains environment, data, training and inference commands

## Final cold-clone smoke test

- [ ] Clone repository into a new directory
- [ ] Pull LFS objects if applicable
- [ ] Create a fresh Python environment
- [ ] Install dependencies only from repository files
- [ ] Download / prepare data using documented commands
- [ ] Run training / evaluation
- [ ] Run inference using produced or downloaded artifacts
