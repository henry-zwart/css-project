#!/bin/bash

# '--quick' option to run reduced analysis
QUICK_OPTION=""
if [[ " $* " == *" --quick "* ]]; then
    QUICK_OPTION="--quick"
fi

echo "Starting Analysis"
echo "Running heatmap_all.py"
uv run scripts/heatmap_all.py $QUICK_OPTION
echo "Running native_only_phase_plots.py"
uv run scripts/native_only_phase_plots.py $QUICK_OPTION
echo "Running plot_simple_finegrained.py"
uv run scripts/plot_simple_finegrained.py $QUICK_OPTION
echo "Running plot_simple_logistic.py"
uv run scripts/plot_simple_logistic.py $QUICK_OPTION
echo "Running proportion_alive_starting_prob.py"
uv run scripts/proportion_alive_starting_prob.py $QUICK_OPTION
echo "Running species_densities_coarsegrained.py"
uv run scripts/species_densities_coarsegrained.py $QUICK_OPTION
echo "Running species_densities_logistic.py"
uv run scripts/species_densities_logistic.py $QUICK_OPTION
echo "Running visualise_vegetation.py"
uv run scripts/visualise_vegetation.py $QUICK_OPTION
echo "Running visualise_veg_invasive.py"
uv run scripts/visualise_veg_invasive.py $QUICK_OPTION
echo "Analysis finished, hooman!"
