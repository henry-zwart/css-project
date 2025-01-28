from css_project.visualisation import invasive_phase_plot
from css_project.logistic import LogisticTwoNative
from css_project.vegetation import InvasiveVegetation


def main():
    MODEL_TYPE = LogisticTwoNative
    WIDTH = 100
    INIT_DENSITY = 0.01
    INITIAL_NUTRIENT_AVAILABILITY = 10
    MIN_NUTRIENT = 5  # 5
    MAX_NUTRIENT = 80  # 80
    P_INV = 0.1
    N_STEPS = 20
    ITERS_PER_STEP = 200  # Bug
    N_REPEATS = 10

    fig, _ = invasive_phase_plot(
        MODEL_TYPE,
        WIDTH,
        INIT_DENSITY,
        INITIAL_NUTRIENT_AVAILABILITY,
        MIN_NUTRIENT,
        MAX_NUTRIENT,
        P_INV,
        N_STEPS,
        ITERS_PER_STEP,
        N_REPEATS,
    )

    fig.savefig("native_invasive_phaseplot.png", dpi=300)


if __name__ == "__main__":
    main()
