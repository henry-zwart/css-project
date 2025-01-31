import argparse

from css_project.fine_grained import FineGrained
from css_project.visualisation import animate_ca, animate_nutrients


def main(width: int, frames: int):
    model = FineGrained(
        width,
        nutrient_level=1.0,
        nutrient_diffusion_rate=0.225,  # 0.225
        nutrient_consume_rate=0.1,
        nutrient_regenerate_rate=0.8,
    )
    model.initial_grid(p=0.1)

    ani = animate_nutrients(model, frames, fps=10)
    ani.save("results/nutrient_diffusion.gif")

    model.reset()
    ani = animate_ca(model, frames, fps=10)
    ani.save("results/fine_grained_vegetation_spread.gif")


if __name__ == "__main__":
    QUICK_WIDTH = 64
    QUICK_FRAMES = 150
    FULL_WIDTH = 256
    FULL_FRAMES = 150

    parser = argparse.ArgumentParser(
        prog="FineGrainedAnimation",
        description="Create animations for finegrained model",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        main(width=QUICK_WIDTH, frames=QUICK_FRAMES)
    else:
        main(width=FULL_WIDTH, frames=FULL_FRAMES)
