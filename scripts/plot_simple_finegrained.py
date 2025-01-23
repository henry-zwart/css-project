import matplotlib.pyplot as plt

from css_project.fine_grained import FineGrained
from css_project.visualisation import animate_nutrients

if __name__ == "__main__":
    width = 128
    model = FineGrained(
        width,
        nutrient_level=1.0,
        nutrient_diffusion_rate=0.0,
        nutrient_consume_rate=0.025,
    )
    model.initial_grid(p=0.1)

    ani = animate_nutrients(model, 100, fps=2)
    # ani = animate_ca(model, 100, fps=2)
    plt.show()
