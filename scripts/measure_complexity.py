import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from css_project.complexity import (
    Compression,
    compressed_size,
    count_clusters,
    fluctuation_cluster_size,
    maximum_cluster_size,
)
from css_project.vegetation import Vegetation
from css_project.visualisation import plot_grid


def main():
    # Prepare states
    WIDTH = 128
    p = 0.01

    weight = (3, 4, 6, 8, 9, 10, 11, 12, 13)
    # weight = (33.5, 33.7, 33.9, 34.1, 34.3, 34.5, 34.7, 34.9, 35.1)
    models = [Vegetation(WIDTH, control=w, alive_prop=p) for w in weight]

    for m in models:
        m.run(1000)

    # Kolmogorov complexity
    complexities = [[] for _ in Compression]
    for m in models:
        for i, compression in enumerate(Compression):
            kc = compressed_size(m.grid, compression)
            complexities[i].append(kc)

    fig, ax = plt.subplots(layout="constrained")
    for i, compression in enumerate(Compression):
        ax.plot(weight, complexities[i], label=compression)
        ax.scatter(weight, complexities[i])
    ax.set_xlabel("Positive weight coefficient")
    ax.set_ylabel("Compressed size")
    ax.set_title("KC estimated as compressed size (50 updates)")
    fig.legend()
    fig.savefig("kolmogorov_complexity.png")

    # Cluster size (maximum and ratio max-min)
    cluster_count = []
    max_clusters = []
    fluctuation_clusters = []
    for m in models:
        cluster_count.append(count_clusters(m.grid) / m.area)
        max_clusters.append(maximum_cluster_size(m.grid))
        fluctuation_clusters.append(fluctuation_cluster_size(m.grid))

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), layout="constrained")
    for ax, model in zip(axes.flatten(), models, strict=True):
        label_matrix, _ = label(model.grid, structure=np.ones((3, 3)))
        ax.imshow(label_matrix)
        ax.set_title(f"{model.positive_factor}")
    fig.savefig(f"islands_{WIDTH}_vegetation.png", dpi=500)

    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(8, 6), layout="constrained"
    )
    axes[0].plot(weight, fluctuation_clusters)
    axes[0].scatter(weight, fluctuation_clusters)
    # axes[0].set_ylim(0, None)
    axes[0].set_yscale("log")
    axes[1].plot(weight, cluster_count)
    axes[1].scatter(weight, cluster_count)
    # axes[1].set_ylim(0, None)
    axes[1].set_yscale("log")
    axes[2].plot(weight, max_clusters)
    axes[2].scatter(weight, max_clusters)
    axes[2].axhline(1, linestyle="dashed", linewidth=1)
    axes[2].axhline(2, linestyle="dashed", linewidth=1)
    # axes[2].set_ylim(0.0, 1.0)
    axes[2].set_yscale("log")
    axes[0].set_ylabel("Cluster size fluctuation")
    axes[1].set_ylabel("Number of clusters")
    axes[2].set_ylabel("Maximum size")
    axes[2].set_xlabel("Positive weight coefficient")
    fig.suptitle("Cluster size complexity for varying +ve weight")
    fig.savefig(f"cluster_complexity_{WIDTH}_vegetation.png", dpi=500)

    # Plot the grids
    fig, axes = plt.subplots(3, 3, figsize=(8, 8), layout="constrained")
    for m, ax in zip(models, axes.flatten(), strict=True):
        _ = plot_grid(m, ax=ax)
        ax.set_title(f"Nutrient availability: {m.positive_factor}")
        # ax.set_title(f"Positive weight: {m.positive_factor}")
    fig.savefig(f"complexity_states_example_{WIDTH}_vegetation.png", dpi=500)


if __name__ == "__main__":
    main()
