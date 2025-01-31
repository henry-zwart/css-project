# Group project for Complex System Simulation

[![Tests](https://github.com/henry-zwart/css-project/actions/workflows/test.yml/badge.svg)](https://github.com/henry-zwart/css-project/actions/workflows/test.yml)

Project repository for [Vegetation Dynamics using Cellular Automata](presentation.pdf), a 
group project completed as part of the 2025 Complex Systems Simulation course at UvA. 

**Documentation:** https://henry-zwart.github.io/css-project/

**Project Board:** https://github.com/users/henry-zwart/projects/1

**Small bonuses:** 
- Our code handles edge cases and invalid parameters.
- PyTest testing suite implemented and succeeds in GitHub CI.
- Our code is structured as a module.
- [Documentation](https://henry-zwart.github.io/css-project/) generated from docstrings

See the [development setup guide](DEVELOPMENT.md) to get started with contributing
to this repository.

## Running code

We have used _uv_ to manage Python and its dependencies. As such, the easiest way 
to reproduce our results and analysis is through uv. 

It is also possible to run the code without uv; however, we do not recommend this 
method, as:
- It has not been thoroughly tested -- we have used uv.
- You must ensure that you are running the correct version of Python, and have the
    correct dependencies installed.

The analysis pipeline is run via the [run_analysis.sh](run_analysis.sh) script. 
After this completes, the resulting figures are located in `results`.

We include a `--quick` flag for convenience, which runs the pipeline in a reduced 
manner. This is effective for verifying code runs successfully, and generates 
graphs with the correct "shape". However, in order to reproduce the results exactly
as seen in our [presentation](presentation.pdf), you should run the 
script without this flag.

### Running with uv (recommended)

Ensure [uv](https://github.com/astral-sh/uv) is installed locally.

uv handles package and Python versioning, ensuring a consistent experiment environment across machines and operating systems.

To reproduce the results execute the following steps:

```zsh
cd path/to/repository

== Run the analysis pipeline
./run_analysis.sh --quick
```


### Running without uv

The experiments rely on Python3.12, and the packages outlined in [requirements.txt](requirements.txt).

> **Note:** If your Python3.12 executable is called something other than `python3.12`, replace this accordingly in the following steps.

To reproduce the results, execute the following steps:

```zsh
cd path/to/repository

# == Create a virtual environment
python3.12 -m venv env

# == Activate the virtual environment
source env/bin/activate # On macOS or Linux
.\env\Scripts\activate # On windows

# == Install package and dependencies
pip install .

# == Run the experiments
ENTRYPOINT=python ./run_analysis.sh --quick
```

## License

This project is open-sourced under an [MIT license](LICENSE.md).
