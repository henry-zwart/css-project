"""
Make phase plots for both coarse-grained and logistic model for multiple control
parameters and initial values.
"""
from css_project.logistic import Logistic
from css_project.vegetation import Vegetation
from css_project.visualisation import logistic_phase_plot


def main():
    fig, _ = logistic_phase_plot(
        Vegetation,
        [64, 128, 256],
        0.1,
        [5, 15],
        min_nutrient=1,
        max_nutrient=20,
        n_steps=50,
        iters_per_step=200,
        n_repeats=20,
        control_variable_name="Positive weight",
    )

    fig.savefig("results/native_vegetation_phaseplot.png", dpi=300)

    fig, _ = logistic_phase_plot(
        Logistic,
        [64, 128, 256],
        0.1,
        # [6.0, 26.0, 55.0, 65.0],
        [15.0, 45.0],
        min_nutrient=1,
        max_nutrient=65,
        n_steps=65,
        iters_per_step=200,
        n_repeats=20,
        control_variable_name="Nutrient availability",
    )

    fig.savefig("results/native_logistic_phaseplot.png", dpi=300)


if __name__ == "__main__":
    main()
