#!/bin/bash

# '--quick' option to run reduced analysis
QUICK_OPTION=""
if [[ " $* " == *" --quick "* ]]; then
    QUICK_OPTION="--quick"
fi


# If python entrypoint not provided, use uv as default
ENTRYPOINT="${ENTRYPOINT:-uv run}"

echo "Starting Analysis"
echo "Running heatmap_all.py"
$ENTRYPOINT scripts/heatmap_all.py $QUICK_OPTION
echo "Running native_only_phase_plots.py"
$ENTRYPOINT scripts/native_only_phase_plots.py $QUICK_OPTION
echo "Running plot_simple_finegrained.py"
$ENTRYPOINT scripts/plot_simple_finegrained.py $QUICK_OPTION
echo "Running plot_simple_logistic.py"
$ENTRYPOINT scripts/plot_simple_logistic.py $QUICK_OPTION
echo "Running proportion_alive_starting_prob.py"
$ENTRYPOINT scripts/proportion_alive_starting_prob.py $QUICK_OPTION
echo "Running species_densities_coarsegrained.py"
$ENTRYPOINT scripts/species_densities_coarsegrained.py $QUICK_OPTION
echo "Running species_densities_logistic.py"
$ENTRYPOINT scripts/species_densities_logistic.py $QUICK_OPTION
echo "Running create_gifs.py"
$ENTRYPOINT scripts/create_gifs.py $QUICK_OPTION
echo "Running plot_invasive_impact.py"
$ENTRYPOINT scripts/plot_invasive_impact.py $QUICK_OPTION
echo "Analysis finished, hooman!"
