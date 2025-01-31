"""
Make phase plots for both coarse-grained and logistic model for multiple control
parameters and initial values.
"""

import argparse

from css_project.logistic import Logistic
from css_project.vegetation import Vegetation
from css_project.visualisation import logistic_phase_plot


def main(widths: list[int], n_repeats: int):
    fig, _ = logistic_phase_plot(
        Vegetation,
        widths,
        0.01,
        [5, 15],
        min_nutrient=1,
        max_nutrient=20,
        n_steps=50,
        iters_per_step=200,
        n_repeats=n_repeats,
        control_variable_name="Positive weight",
    )

    fig.savefig("results/native_vegetation_phaseplot.png", dpi=300)

    fig, _ = logistic_phase_plot(
        Logistic,
        widths,
        0.01,
        [15.0, 45.0],
        min_nutrient=1,
        max_nutrient=65,
        n_steps=65,
        iters_per_step=200,
        n_repeats=n_repeats,
        control_variable_name="Nutrient availability",
    )

    fig.savefig("results/native_logistic_phaseplot.png", dpi=300)


if __name__ == "__main__":
    QUICK_WIDTHS = [32, 64, 128]
    QUICK_REPEATS = 2
    FULL_WIDTHS = [64, 128, 256]
    FULL_REPEATS = 20

    parser = argparse.ArgumentParser(
        prog="PhaseTransitions", description="Create continuous phase transition plots"
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        main(widths=QUICK_WIDTHS, n_repeats=QUICK_REPEATS)
    else:
        main(widths=FULL_WIDTHS, n_repeats=FULL_REPEATS)
