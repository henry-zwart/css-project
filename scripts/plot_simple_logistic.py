"""Make an animation of the vegetation over time of the logistic model."""

import argparse

from tqdm import trange

from css_project.logistic import LogisticTwoNative
from css_project.visualisation import animate_ca


def main(width: int, frames: int):
    model = LogisticTwoNative(
        width,
        consume_rate_1=63.7,
        consume_rate_2=63.7 * (0.5),
        control=35,
        species_prop=[0.001, 0.0],
    )
    for _ in trange(500):
        model.update()

    # model.introduce_invasive(1.0)
    ani = animate_ca(model, frames, fps=2)
    ani.save("results/logistic_two_species.gif")


if __name__ == "__main__":
    QUICK_WIDTH = 128
    QUICK_FRAMES = 20
    FULL_WIDTH = 256
    FULL_FRAMES = 20

    parser = argparse.ArgumentParser(
        prog="LogisticPlot",
        description="Create animation of logistic model",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        main(QUICK_WIDTH, QUICK_FRAMES)
    else:
        main(FULL_WIDTH, FULL_FRAMES)
