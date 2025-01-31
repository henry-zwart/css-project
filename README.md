# Group project for Complex System Simulation

[![Tests](https://github.com/henry-zwart/css-project/actions/workflows/test.yml/badge.svg)](https://github.com/henry-zwart/css-project/actions/workflows/test.yml)

**Documentation:** https://henry-zwart.github.io/css-project/

**Small bonuses:** 
- Our code handles edge cases and invalid parameters.
- Our "pytest" works and succeeds.
- Our code is structured as a module.
- Documentation generated from docstrings (see link above).

## Development - Getting started

This repository uses _uv_ to manage Python and its dependencies, and _pre-commit_ to run
automatic code linting & formatting.

1. Install [uv](https://github.com/astral-sh/uv)

2. Navigate to this project directory

3. Install pre-commit:

```zsh
# We can use uv to install pre-commit!
uv tool install pre-commit --with pre-commit-uv --force-reinstall

# Check that pre-commit installed alright (should say 3.8.0 or similar)
pre-commit --version

# After installing pre-commit, you'll need to initialise it.
# This installs all pre-commit hooks, the scripts which run before a commit.
pre-commit install

# It's a good idea to run pre-commit now on all files.
pre-commit run --all-files
```

4. Run code:

```zsh
uv run <PATH-TO-PYTHON-FILE>
```

## Adding new packages with uv

It's simple to add a Python package to the project with uv using `uv add`.
For instance, to install numpy and scipy, and add them to the dependency graph, run:

```zsh
uv add numpy scipy
```

To remove a package dependency, e.g., scipy, run:

```zsh
uv remove scipy
```

## Running JupyterLab with uv

uv can also be used to run Jupyter, the same way as we run Python scripts:

```zsh
# If Jupyter Lab isn't already installed, add it as a dependency
uv add jupyterlab

# Start a JupyterLab server
uv run jupyter lab
```

## Pre-commit hooks

I've included a couple of pre-commit hooks 
(see [.pre-commit-config.yaml](.pre-commit-config.yaml)) which will be executed every 
time we commit code to git. Both pre-commit hooks come from the 
[ruff](https://github.com/astral-sh/ruff) Python linter:
- `ruff`: lints Python code to ensure it adheres to the PEP8 standards. Includes a bunch of nice things like automatic sorting of imports by name and type.
- `ruff format`: formats Python code in a consistent manner, including removing excess whitespace, swapping tabs for spaces, and a _tonne_ of other things.

These should help to keep code tidy and consistent, with no extra effort on our part. 
Both of the hooks should run automagically if you've followed the setup instructions for
[installing pre-commit with uv](#development---getting-started).

## License

This project is open-sourced under an [MIT license](LICENSE.md).
