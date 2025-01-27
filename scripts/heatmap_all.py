import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from css_project.complexity import (
    Compression,
    compressed_size,
    count_clusters,
    maximum_cluster_size,
    ratio_cluster_size,
)
from css_project.vegetation import Vegetation


def main():
    # Prepare states
    WIDTH = 64

    positives = np.linspace(start=1, stop=20, num=20)
    initial_probs = np.logspace(start=-3, stop=0, num=10)
    models = [
        [
            Vegetation(WIDTH, positive_factor=positive, alive_prop=p)
            for p in initial_probs
        ]
        for positive in positives
    ]

    for i in range(len(positives)):
        for m in tqdm(models[i]):
            m.find_steady_state(100)

    alive_grid = []
    for i in range(len(positives)):
        prob_results = []
        for m in models[i]:
            prob_results.append(m.proportion_alive_list[-1])
        alive_grid.append(prob_results)

    complexity_grid = []
    for i in range(len(positives)):
        prob_results = []
        for m in models[i]:
            kc = compressed_size(m.grid, Compression.ZLIB)
            prob_results.append(kc)
        complexity_grid.append(prob_results)

    # Cluster size (maximum and ratio max-min)
    cluster_count = []
    max_clusters = []
    ratio_clusters = []
    for i in range(len(positives)):
        prob_cluster_count = []
        prob_max_clusters = []
        prob_ratio_clusters = []
        for m in models[i]:
            prob_cluster_count.append(count_clusters(m.grid))
            prob_max_clusters.append(maximum_cluster_size(m.grid))
            prob_ratio_clusters.append(ratio_cluster_size(m.grid))

        cluster_count.append(prob_cluster_count)
        max_clusters.append(prob_max_clusters)
        ratio_clusters.append(prob_ratio_clusters)

    print(ratio_clusters)
    alive_grid = np.array(alive_grid)
    cluster_count = np.array(cluster_count)
    max_clusters = np.array(max_clusters)
    ratio_clusters = np.array(ratio_clusters)
    ratio_clusters[ratio_clusters == np.inf] = np.nan

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), layout="constrained")
    sns.heatmap(
        alive_grid, cmap="viridis", xticklabels=[], yticklabels=positives, ax=axes[0, 0]
    )
    sns.heatmap(cluster_count, cmap="viridis", ax=axes[0, 1])
    sns.heatmap(max_clusters, cmap="viridis", ax=axes[1, 0])
    sns.heatmap(ratio_clusters, cmap="viridis", ax=axes[1, 1])

    axes[0, 0].set_title("Cell density")
    axes[0, 1].set_title("Cluster count")
    axes[1, 0].set_title("Maximum size of a cluster")
    axes[1, 1].set_title("Size ratio max cluster/min cluster")

    fig.savefig("heatmaps_all_native.png", dpi=500)


if __name__ == "__main__":
    main()
