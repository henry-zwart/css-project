from css_project.logistic import Logistic
from css_project.vegetation import Vegetation
from css_project.visualisation import logistic_phase_plot


def main():
    fig, _ = logistic_phase_plot(
        Vegetation,
        128,
        0.01,
        [6, 10, 15],
        min_nutrient=5,
        max_nutrient=20,
        n_steps=20,
        iters_per_step=200,
        n_repeats=10,
    )

    fig.savefig("results/native_vegetation_phaseplot.png", dpi=300)

    fig, _ = logistic_phase_plot(
        Logistic,
        128,
        0.01,
        [6.0, 26.0, 55.0, 65.0],
        min_nutrient=1,
        max_nutrient=80,
        n_steps=80,
        iters_per_step=200,
        n_repeats=10,
    )

    fig.savefig("results/native_logistic_phaseplot.png", dpi=300)


if __name__ == "__main__":
    main()
