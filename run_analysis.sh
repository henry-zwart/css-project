#!/bin/bash

echo "Starting Analysis"
echo "Running heatmap_all.py"
uv run scripts/heatmap_all.py
echo "Running native_only_phase_plots.py"
uv run scripts/native_only_phase_plots.py
echo "Running plot_simple_finegrained.py"
uv run scripts/plot_simple_finegrained.py
echo "Running plot_simple_logistic.py"
uv run scripts/plot_simple_logistic.py
echo "Running proportion_alive_starting_prob.py"
uv run scripts/proportion_alive_starting_prob.py
echo "Running species_densities_coarsegrained.py"
uv run scripts/species_densities_coarsegrained.py
echo "Running species_densities_logistic.py"
uv run scripts/species_densities_logistic.py
echo "Running visualise_vegetation.py"
uv run scripts/visualise_vegetation.py
echo "Running visualise_veg_invasive.py"
uv run scripts/visualise_veg_invasive.py
echo "Analysis finished, hooman!"
