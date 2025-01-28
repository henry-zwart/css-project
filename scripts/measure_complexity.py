import matplotlib.pyplot as plt

from css_project.complexity import (
    Compression,
    avg_cluster_size,
    compressed_size,
    count_clusters,
    fluctuation_cluster_size,
    maximum_cluster_size,
    ratio_cluster_size,
    variance_cluster_size,
)
from css_project.logistic import Logistic
from css_project.model import VegetationModel
from css_project.vegetation import Vegetation
from css_project.visualisation import plot_grid


def plot_cluster_complexity(models: list[VegetationModel], values):
    cluster_count = []
    max_clusters = []
    ratio_clusters = []
    mean_clusters = []
    variance_clusters = []
    fluctuation_clusters = []

    for m in models:
        cluster_count.append(count_clusters(m.grid))
        max_clusters.append(maximum_cluster_size(m.grid))
        ratio_clusters.append(ratio_cluster_size(m.grid))
        mean_clusters.append(avg_cluster_size(m.grid))
        variance_clusters.append(variance_cluster_size(m.grid))
        fluctuation_clusters.append(fluctuation_cluster_size(m.grid))

    # Plot complexity measures
    fig, axes = plt.subplots(
        nrows=6, ncols=1, sharex=True, figsize=(8, 10), layout="constrained"
    )
    axes[0].plot(values, ratio_clusters)
    axes[0].scatter(values, ratio_clusters)
    axes[0].set_ylim(0, None)
    axes[1].plot(values, cluster_count)
    axes[1].scatter(values, cluster_count)
    axes[1].set_ylim(0, None)
    axes[2].plot(values, max_clusters)
    axes[2].scatter(values, max_clusters)
    axes[2].set_ylim(0.0, 1.0)
    axes[3].plot(values, mean_clusters)
    axes[4].plot(values, variance_clusters)
    axes[5].plot(values, fluctuation_clusters)
    axes[0].set_ylabel("Max/min size")
    axes[1].set_ylabel("Number of clusters")
    axes[2].set_ylabel("Maximum size")
    axes[3].set_ylabel("Mean size")
    axes[4].set_ylabel("Size variance")
    axes[5].set_ylabel("Size fluctuation")
    axes[5].set_xlabel("Control parameter")
    fig.suptitle("Cluster size complexity for varying control parameter")

    # Plot the grids
    fig_examples, axes_examples = plt.subplots(
        3, 3, figsize=(8, 8), layout="constrained"
    )
    for m, v, ax in zip(models, values, axes_examples.flatten(), strict=True):
        _ = plot_grid(m, ax=ax)
        ax.set_title(f"Control parameter: {v}")

    return fig, fig_examples


def main():
    # Prepare states
    WIDTH = 64
    WIDTH = 256
    cg_p = 0.01
    lo_p = 0.001

    weight = (3, 4, 6, 8, 9, 10, 11, 13, 17)
    nutrient = (1, 5, 10, 15, 25, 35, 45, 55, 65)
    cg_models = [Vegetation(WIDTH, control=w, alive_prop=cg_p) for w in weight]
    log_models = [Logistic(WIDTH, control=v, alive_prop=lo_p) for v in nutrient]

    for m in cg_models:
        m.run(1000)

    for m in log_models:
        m.run(1000)

    # Kolmogorov complexity
    complexities = [[] for _ in Compression]
    for m in cg_models:
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

    # Coarse-grained model
    complexities_fig, examples_fig = plot_cluster_complexity(cg_models, weight)
    complexities_fig.savefig("cg_cluster_complexity.png", dpi=500)
    examples_fig.savefig("cg_complexity_examples.png", dpi=500)

    # Logistic model
    complexities_fig, examples_fig = plot_cluster_complexity(log_models, nutrient)
    complexities_fig.savefig("logistic_cluster_complexity.png", dpi=500)
    examples_fig.savefig("logistic_complexity_examples.png", dpi=500)


if __name__ == "__main__":
    main()
