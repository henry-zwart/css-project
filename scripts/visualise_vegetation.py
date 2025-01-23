import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, sum

from css_project.vegetation import InvasiveVegetation, Vegetation
from css_project.visualisation import animate_ca, plot_grid

if __name__ == "__main__":
    timespan = 100
    width = 256

    # This code runs without invasive species
    """vegetation = Vegetation(width)
    vegetation.initial_grid(p=0.5)

    initial_grid = vegetation.grid.copy()

    ani = animate_ca(vegetation, 20)
    ani.save("vegetation.gif")

    vegetation.grid = initial_grid.copy()

    fig, ax = plot_grid(vegetation)
    fig.savefig("veg_grid.png", dpi=300)

    t = 0
    total_cells = width * width
    alive = [vegetation.total_alive() / total_cells]

    while t < 20:
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

    alive_nat = []
    alive_inv = []
    alive_n, alive_i = vegetation.total_alive()
    alive_nat.append(alive_n / total_cells)
    alive_inv.append(alive_i / total_cells)

    while t < timespan:
        vegetation.update()
        alive_n, alive_i = vegetation.total_alive()
        alive_nat.append(alive_n / total_cells)
        alive_inv.append(alive_i / total_cells)
        t += 1

    fig, ax = plot_grid(vegetation)
    fig.savefig("veg_grid_after.png", dpi=300)

    # Reset grid to initial state
    vegetation.grid = initial_grid.copy()

    # Make animation of grid
    ani = animate_ca(vegetation, timespan)
    ani.save("vegetation.gif")

    # Plot ratio of dead, native, and invasive cells
    iterations = list(range(len(alive_nat)))

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, alive_nat, linestyle="-", label="Native Species")
    plt.plot(iterations, alive_inv, linestyle="-", label="Invasive Species")
    plt.title("Proportion of Native and Invasive species vs Iterations")
    plt.xlabel("Time Step")
    plt.ylabel("Proportion Cells")
    plt.savefig("proportion_nat_inv.png", dpi=300)

    # Cluster size distribution

    # Identify each cluster in the matrix
    cluster_matrix, cluster_count = label(
        vegetation.grid, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    )

    # Find the sizes of the clusters
    area = sum(
        vegetation.grid, cluster_matrix, index=np.arange(cluster_matrix.max() + 1)
    )

    clusters = area[1:]

    # Find frequencies of the found cluster sizes
    clusters_freq = np.unique(clusters, return_counts=True)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(clusters_freq[0], clusters_freq[1], "o")
    plt.xlim([6, 1000])
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Cluster size distribution")
    plt.xlabel("Cluster size")
    plt.ylabel("Number of patches")
    plt.savefig("cluster_size_distribution.png", dpi=300)
    plt.show()
