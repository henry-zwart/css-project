from css_project.visualisation import logistic_phase_plot


def main():
    WIDTH = 100
    INIT_DENSITY = 0.01
    INITIAL_NUTRIENT_AVAILABILITY = 10
    MIN_NUTRIENT = 5  # 5
    MAX_NUTRIENT = 80  # 80
    N_STEPS = 80
    ITERS_PER_STEP = 200  # Bug
    N_REPEATS = 10

    fig, _ = logistic_phase_plot(
        WIDTH,
        INIT_DENSITY,
        INITIAL_NUTRIENT_AVAILABILITY,
        MIN_NUTRIENT,
        MAX_NUTRIENT,
        N_STEPS,
        ITERS_PER_STEP,
        N_REPEATS,
    )

    fig.savefig("native_phaseplot.png", dpi=300)


if __name__ == "__main__":
    main()
