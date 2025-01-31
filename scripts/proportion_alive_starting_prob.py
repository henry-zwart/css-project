"""
Make a figure of the equilibrium density depending on the
initial probability for multiple control values.

Make a figure of the proportion alive over time for multiple
initial probabilities.
"""

import argparse

import numpy as np

from css_project.visualisation import phase_transition_prob


def main(width: int, resolution: int):
    p_list = np.logspace(start=-4, stop=0, num=resolution, base=10)
    width = width
    pos_weight_list = [5, 7, 9]

    fig1, fig2 = phase_transition_prob(width, p_list, pos_weight_list)

    fig1.savefig("results/proportion_alive_over_iter", dpi=300)
    fig2.savefig("results/proportion_alive_on_probability", dpi=300)


if __name__ == "__main__":
    QUICK_WIDTH = 64
    QUICK_RESOLUTION = 20
    FULL_WIDTH = 64
    FULL_RESOLUTION = 100

    parser = argparse.ArgumentParser(
        prog="EquilibriumDensity",
        description="Create plot of equilibrium density given initial density",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        main(width=QUICK_WIDTH, resolution=QUICK_RESOLUTION)
    else:
        main(width=FULL_WIDTH, resolution=FULL_RESOLUTION)
