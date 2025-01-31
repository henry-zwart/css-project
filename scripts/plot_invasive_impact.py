import argparse

from css_project.visualisation import eq_after_inv


def main(width: int, resolution: int, n_repeats: int):
    p_nat = 0.1

    # Spots
    control = 5
    fig = eq_after_inv(width, p_nat, control, n_repeats, resolution)
    fig.savefig("invasive_impact_spots.png", dpi=300)

    # Labyrinth
    control = 9.5
    fig = eq_after_inv(width, p_nat, control, n_repeats, resolution)
    fig.savefig("invasive_impact_labyrinth.png", dpi=300)

    # Giant component
    control = 10.5
    fig = eq_after_inv(width, p_nat, control, n_repeats, resolution)
    fig.savefig("invasive_impact_gc.png", dpi=300)

    # Dense
    control = 15
    fig = eq_after_inv(width, p_nat, control, n_repeats, resolution)
    fig.savefig("invasive_impact_dense.png", dpi=300)


if __name__ == "__main__":
    QUICK_WIDTH = 64
    QUICK_REPEATS = 2
    QUICK_RESOLUTION = 10
    FULL_WIDTH = 128
    FULL_REPEATS = 5
    FULL_RESOLUTION = 100

    parser = argparse.ArgumentParser(
        prog="InvasiveImpact",
        description=(
            "Plot the impact of varying invasive proportion on "
            "native equilibrium population"
        ),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        main(width=QUICK_WIDTH, n_repeats=QUICK_REPEATS, resolution=QUICK_RESOLUTION)
    else:
        main(width=FULL_WIDTH, n_repeats=FULL_REPEATS, resolution=FULL_RESOLUTION)
