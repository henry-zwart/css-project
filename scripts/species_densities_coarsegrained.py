"""
Visualise species densities over time for coarse-grained model.
"""

import argparse

from css_project.visualisation import densities_invasive_coarsegrained


def main(width: int):
    p_nat = 0.25
    p_inv = 0.6

    fig1 = densities_invasive_coarsegrained(width, p_nat, p_inv)

    fig1.savefig("results/species_densities_coarsegrained", dpi=300)


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
