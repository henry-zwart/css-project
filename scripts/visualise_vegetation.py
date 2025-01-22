import matplotlib.pyplot as plt

from css_project.vegetation import InvasiveVegetation, Vegetation
from css_project.visualisation import animate_ca, plot_grid

if __name__ == "__main__":
    width = 128

    # This code runs without invasive species
    """vegetation = Vegetation(width)
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

    fig, ax = plot_grid(vegetation)
    fig.savefig("veg_grid_after.png", dpi=300)
    
    # Plot ratio of alive vs dead cells
    # vegetation = Vegetation(128)
    # vegetation.initial_grid(p=0.5)
    # ani = animate_ca(vegetation, 1)
    # plt.show()
    # ani.save("vegetation.mp4")
    # """

    # This code runs with invasive species

    vegetation = InvasiveVegetation(width)
    vegetation.initial_grid()

    # Copy grid to use in graph
    initial_grid = vegetation.grid.copy()

    fig, ax = plot_grid(vegetation)
    fig.savefig("veg_grid.png", dpi=300)

    t = 0
    total_cells = width * width
    # alive_nat, alive_inv = [vegetation.total_alive()]

    while t < 40:
        vegetation.update()
        # alive_n, alive_i = [vegetation.total_alive()]
        # alive_nat.append(alive_n)
        # alive_inv.append(alive_i)
        t += 1

    fig, ax = plot_grid(vegetation)
    fig.savefig("veg_grid_after.png", dpi=300)

    # Reset grid to initial state
    vegetation.grid = initial_grid.copy()

    ani = animate_ca(vegetation, 100)
    ani.save("vegetation.gif")
