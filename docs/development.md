# Development Guide

Install from the repository so local code edits are used:

```sh
micromamba env create -f environment.yml
micromamba activate twopy
micromamba run -n twopy pre-commit install
```

The development environment installs twopy as an editable package, so the
`twopy` terminal command is available after activating the environment. If the
environment already existed before the command was added, refresh the editable
install:

```sh
micromamba run -n twopy python -m pip install -e .
```

Run the full gate before handoff:

```sh
micromamba run -n twopy pre-commit run --all-files
```

The installed pre-commit hook runs ruff, ty, and unit tests before each commit.
