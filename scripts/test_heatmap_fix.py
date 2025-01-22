import matplotlib.pyplot as plt

from css_project.vegetation import Vegetation
from css_project.visualisation import plot_grid

if __name__ == "__main__":
    model = Vegetation(width=64)
    fig, ax = plot_grid(model)
    plt.show()

    model.initial_grid(0.5)
    fig, ax = plot_grid(model)
    plt.show()

    model.grid[:, :] = 1
    fig, ax = plot_grid(model)
    plt.show()
