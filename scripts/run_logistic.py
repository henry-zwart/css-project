"""Visualise logistic model as animation or image."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from css_project.logistic import Logistic
from css_project.visualisation import animate_ca, plot_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Logistic", description="Run logistic model")
    parser.add_argument("--width", type=int)
    parser.add_argument("--init-density", type=float)
    parser.add_argument("--control", type=float)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--frames", type=int, nargs="?", const=600)
    parser.add_argument("--fps", type=int, nargs="?", const=30)
    parser.add_argument("--save-path", type=Path, nargs="?", const=None)
    args = parser.parse_args()

    model = Logistic(
        args.width,
        control=args.control,
        alive_prop=args.init_density,
    )

    if args.animate:
        ani = animate_ca(model, args.frames, args.fps)
        if args.save_path:
            ani.save(args.save_path)
    else:
        model.run(1000)
        fig, _ = plot_grid(model)
        if args.save_path and fig is not None:
            fig.savefig(args.save_path, dpi=800)

    if not args.save_path:
        plt.show()
