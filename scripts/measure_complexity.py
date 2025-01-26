import matplotlib.pyplot as plt
from tqdm import trange

from css_project.complexity import (
    Compression,
    compressed_size,
    count_clusters,
    maximum_cluster_size,
    ratio_cluster_size,
)
from css_project.vegetation import Vegetation
from css_project.visualisation import plot_grid


def main():
    # Prepare states
    WIDTH = 64
    p = 0.5
    weight = (3, 4, 6, 8, 9, 10, 11, 13, 17)
    models = [Vegetation(WIDTH, positive_factor=w) for w in weight]
    for m in models:
        m.initial_grid(p)
        for _ in trange(50):
            m.update()

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
    ratio_clusters = []
    for m in models:
        cluster_count.append(count_clusters(m.grid))
        max_clusters.append(maximum_cluster_size(m.grid))
        ratio_clusters.append(ratio_cluster_size(m.grid))

    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(8, 6), layout="constrained"
    )
    axes[0].plot(weight, ratio_clusters)
    axes[0].scatter(weight, ratio_clusters)
    axes[0].set_ylim(0, None)
    axes[1].plot(weight, cluster_count)
    axes[1].scatter(weight, cluster_count)
    axes[1].set_ylim(0, None)
    axes[2].plot(weight, max_clusters)
    axes[2].scatter(weight, max_clusters)
    axes[2].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Max/min size")
    axes[1].set_ylabel("Number of clusters")
    axes[2].set_ylabel("Maximum size")
    axes[2].set_xlabel("Positive weight coefficient")
    fig.suptitle("Cluster size complexity for varying +ve weight")
    fig.savefig("cluster_complexity.png", dpi=500)

    # Plot the grids
    fig, axes = plt.subplots(3, 3, figsize=(8, 8), layout="constrained")
    for m, ax in zip(models, axes.flatten(), strict=True):
        _ = plot_grid(m, ax=ax)
        ax.set_title(f"Positive weight: {m.positive_factor}")
    fig.savefig("complexity_states_example.png", dpi=500)


if __name__ == "__main__":
    main()
