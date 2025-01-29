import argparse

import matplotlib.pyplot as plt

from css_project.vegetation import Vegetation
from css_project.visualisation import animate_ca, plot_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Logistic", description="Run logistic model")
    parser.add_argument("--width", type=int)
    parser.add_argument("--init-density", type=float)
    parser.add_argument("--control", type=float)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--frames", type=int, nargs="?", const=600)
    parser.add_argument("--fps", type=int, nargs="?", const=30)
    args = parser.parse_args()

    model = Vegetation(
        args.width,
        control=args.control,
        alive_prop=args.init_density,
    )

    if args.animate:
        ani = animate_ca(model, args.frames, args.fps)
    else:
        model.run(1000)
        plot_grid(model)
    plt.show()
