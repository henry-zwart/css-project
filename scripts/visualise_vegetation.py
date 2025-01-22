import matplotlib.pyplot as plt

from css_project.vegetation import Vegetation
from css_project.visualisation import animate_ca, plot_grid

if __name__ == "__main__":
    width = 64

    vegetation = Vegetation(width)
    vegetation.initial_grid(p=0.5)

    initial_grid = vegetation.grid.copy()

    ani = animate_ca(vegetation, 10)
    ani.save("vegetation.gif")

    vegetation.grid = initial_grid.copy()

    fig, ax = plot_grid(vegetation)
    fig.savefig("veg_grid.png", dpi=300)

    t = 0
    total_cells = width * width
    alive = [vegetation.total_alive() / total_cells]

    while t < 10:
        vegetation.update()
        alive.append(vegetation.total_alive() / total_cells)
        t += 1

    print(vegetation.total_alive())
    fig, ax = plot_grid(vegetation)
    fig.savefig("veg_grid_after.png", dpi=300)

    iterations = list(range(len(alive)))

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, alive, linestyle="-")
    plt.title("Proportion of Alive Plants vs Iterations")
    plt.xlabel("Time Step")
    plt.ylabel("Proportion of Alive Plants")
    plt.savefig("proportion_alive.png", dpi=300)
