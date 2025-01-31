"""Make gifs of the coarse-grained model and the logistic model without
invasive species for three values of control parameters."""

import argparse

from tqdm import tqdm

from css_project.logistic import LogisticTwoNative
from css_project.vegetation import Vegetation
from css_project.visualisation import animate_ca


def main(width: int, frames_logistic: int):
    pos_list = [6, 9, 13]
    control_list = [10, 30, 50]

    # This code makes gifs of activator-inhibitor model for multiple control parameters
    with tqdm(total=len(pos_list) + len(control_list)) as prog:
        for pos in pos_list:
            vegetation = Vegetation(width, control=pos)
            vegetation.initial_grid(p=0.3)

            ani = animate_ca(vegetation, 10)
            ani.save(f"results/activator_inhibitor_control_{pos}.gif")
            prog.update()

        for control_val in control_list:
            model = LogisticTwoNative(
                width,
                consume_rate_1=63.7,
                control=control_val,
                species_prop=[0.001, 0.0],
            )
            ani = animate_ca(model, frames_logistic, fps=30)
            ani.save(f"results/logistic_control_{control_val}.gif")
            prog.update()


if __name__ == "__main__":
    WIDTH = 64
    QUICK_FRAMES_LOGISTIC = 10
    FULL_FRAMES_LOGISTIC = 1000

    parser = argparse.ArgumentParser(
        prog="Gifs",
        description=(
            "Generate GIFS of Activator-Inhibitor and "
            "Logistic models for presentation"
        ),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        main(width=WIDTH, frames_logistic=QUICK_FRAMES_LOGISTIC)
    else:
        main(width=WIDTH, frames_logistic=FULL_FRAMES_LOGISTIC)
