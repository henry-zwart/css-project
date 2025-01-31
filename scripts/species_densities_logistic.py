"""
Visualise species densities over time for logistic model.
"""

import argparse

from css_project.visualisation import densities_invasive_logistic


def main(width: int):
    random_seed = 1
    p = 0.9

    fig1 = densities_invasive_logistic(width, random_seed, p)

    fig1.savefig("results/species_densities_logistic", dpi=300)


if __name__ == "__main__":
    QUICK_WIDTH = 128
    FULL_WIDTH = 256

    parser = argparse.ArgumentParser(
        prog="SpeciesDensityEvolution",
        description="Plot evolution of species densities after invasive introduction",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        main(width=QUICK_WIDTH)
    else:
        main(width=FULL_WIDTH)
