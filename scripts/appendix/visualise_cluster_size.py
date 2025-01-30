import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from scipy.ndimage import label, sum
from tqdm import trange

from css_project.vegetation import Vegetation

if __name__ == "__main__":
    timespan = 100
    width = 256

    # This code runs without invasive species
    vegetation = Vegetation(width)
    vegetation.initial_grid(p=0.1)
    vegetation.positive_factor = 8

    initial_grid = vegetation.grid.copy()

    # ani = animate_ca(vegetation, 20)
    # ani.save("vegetation.gif")

    vegetation.grid = initial_grid.copy()
    n = 50
    total_cells = width * width
    alive = [vegetation.total_alive() / total_cells]
    for _ in trange(n):
        vegetation.update()
        alive.append(vegetation.total_alive() / total_cells)

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

    # Plot of cluster size distribution
    plt.figure(figsize=(8, 6))
    plt.plot(clusters_freq[0], clusters_freq[1], "o")
    plt.xlim([6, 1000])
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Cluster size distribution")
    plt.xlabel("Cluster size")
    plt.ylabel("Number of patches")
    plt.savefig("appendix_results/cluster_size_distribution.png", dpi=300)

    # Use powerlaw package to fit the data
    results = powerlaw.Fit(clusters)
    print(results.power_law.alpha)
    print(results.power_law.xmin)
    # Plot of probability density function and power law
    plt.figure(figsize=(8, 6))
    results.plot_pdf(label="Cluster Size Data")
    results.power_law.plot_pdf(linestyle="--", label="Power Law Fit")
    plt.legend()
    plt.xlabel("Cluster Size")
    plt.ylabel("Probability Density")
    plt.title("Cluster Size Power Law")
    plt.savefig("appendix_results/cluster_size_power_law.png", dpi=300)
