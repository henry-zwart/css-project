import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from css_project.complexity import (
    Compression,
    compressed_size,
)
from css_project.vegetation import Vegetation


def main():
    # Prepare states
    WIDTH = 64

    positives = np.linspace(start=1, stop=20, num=20)
    initial_probs = np.logspace(start=-3, stop=0, num=200)
    models = [
        [
            Vegetation(WIDTH, positive_factor=positive, alive_prop=p)
            for p in initial_probs
        ]
        for positive in positives
    ]

    for i in range(len(positives)):
        for m in tqdm(models[i]):
            m.find_steady_state(200)

    complexity_grid = []
    for i in range(len(positives)):
        prob_results = []
        for m in models[i]:
            kc = compressed_size(m.grid, Compression.ZLIB)
            prob_results.append(kc)
        complexity_grid.append(prob_results)

    # Cluster size (maximum and ratio max-min)
    complexity_grid = np.array(complexity_grid)

    sns.heatmap(complexity_grid, cmap="viridis", xticklabels=[], yticklabels=positives)

    plt.savefig("kolmogorov_heatmap.png", dpi=500)


if __name__ == "__main__":
    main()
