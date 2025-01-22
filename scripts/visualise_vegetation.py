from css_project.vegetation import Vegetation
from css_project.visualisation import plot_grid

if __name__ == "__main__":
    width = 64

    vegetation = Vegetation(width)
    vegetation.initial_grid(p=0.5)
    fig, ax = plot_grid(vegetation)
    fig.savefig("veg_grid.png", dpi=300)

    t = 0
    total_cells = width * width
    alive = [vegetation.total_alive()]

    while t < 40:
        vegetation.update()
        alive.append(vegetation.total_alive())
        t += 1

    fig, ax = plot_grid(vegetation)
    fig.savefig("veg_grid_after.png", dpi=300)

    # Plot ratio of alive vs dead cells
    # vegetation = Vegetation(128)
    # vegetation.initial_grid(p=0.5)
    # ani = animate_ca(vegetation, 1)
    # plt.show()
    # ani.save("vegetation.mp4")
