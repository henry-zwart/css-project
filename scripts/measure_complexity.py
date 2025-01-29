from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from css_project.complexity import (
    Compression,
    avg_cluster_size,
    compressed_size,
    count_clusters,
    fluctuation_cluster_size,
    maximum_cluster_size,
)
from css_project.logistic import Logistic
from css_project.model import VegetationModel
from css_project.vegetation import Vegetation
from css_project.visualisation import plot_grid


def plot_cluster_complexity(models: Sequence[VegetationModel], values):
    cluster_count = []
    max_clusters = []
    fluctuation_clusters = []

    for m in models:
        cluster_count.append(count_clusters(m.grid))
        max_clusters.append(maximum_cluster_size(m.grid))
        fluctuation_clusters.append(fluctuation_cluster_size(m.grid))

    # Plot complexity measures
    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(8, 10), layout="constrained"
    )
    axes[0].plot(values, fluctuation_clusters)
    axes[0].scatter(values, fluctuation_clusters)
    axes[0].set_yscale("log")
    axes[1].plot(values, cluster_count)
    axes[1].scatter(values, cluster_count)
    axes[1].set_yscale("log")
    axes[2].plot(values, max_clusters)
    axes[2].scatter(values, max_clusters)
    axes[2].axhline(1, linestyle="dashed", linewidth=1)
    axes[2].axhline(2, linestyle="dashed", linewidth=1)
    axes[2].set_yscale("log")
    axes[0].set_ylabel("Cluster size fluctuation")
    axes[1].set_ylabel("Number of clusters")
    axes[2].set_ylabel("Maximum size")
    axes[2].set_xlabel("Control parameter value")
    fig.suptitle("Cluster size complexity for varying control parameter")

    # Plot the grids
    fig_examples, axes_examples = plt.subplots(
        3, 3, figsize=(8, 8), layout="constrained"
    )
    for m, v, ax in zip(models, values, axes_examples.flatten(), strict=True):
        _ = plot_grid(m, ax=ax)
        ax.set_title(f"Control parameter: {v}")

    fig_islands, axes_islands = plt.subplots(
        nrows=3, ncols=3, figsize=(8, 8), layout="constrained"
    )
    for ax, model in zip(axes_islands.flatten(), models, strict=True):
        label_matrix, _ = label(model.grid, structure=np.ones((3, 3)))
        ax.imshow(label_matrix)
        try:
            ax.set_title(f"{model.positive_factor}")
        except AttributeError:
            ax.set_title(f"{model.supplement_rate}")
    # fig.savefig(f"islands_{width}_{model_type}.png", dpi=500)

    return fig, fig_examples, fig_islands


def main():
    # Prepare states
    WIDTH = 64
    WIDTH = 128
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
    complexities_fig, examples_fig, islands_fig = plot_cluster_complexity(
        cg_models, weight
    )
    complexities_fig.savefig(f"cg_cluster_complexity_{WIDTH}.png", dpi=500)
    examples_fig.savefig(f"cg_complexity_examples_{WIDTH}.png", dpi=500)
    islands_fig.savefig(f"cg_islands_{WIDTH}.png", dpi=500)

    # Logistic model
    complexities_fig, examples_fig, islands_fig = plot_cluster_complexity(
        log_models, nutrient
    )
    complexities_fig.savefig(f"logistic_cluster_complexity_{WIDTH}.png", dpi=500)
    examples_fig.savefig(f"logistic_complexity_examples_{WIDTH}.png", dpi=500)
    islands_fig.savefig(f"logistic_islands_{WIDTH}.png", dpi=500)


if __name__ == "__main__":
    main()
