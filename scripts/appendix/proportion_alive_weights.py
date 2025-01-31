"""Proportion of alive cells based on control variable"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from css_project.vegetation import Vegetation

# from css_project.visualisation import animate_ca, plot_grid
positives = np.linspace(start=1, stop=20, num=20)
negatives = np.linspace(start=1, stop=10, num=10)
width = 64
p = 0.2
alive_grid = []

for positive in tqdm(positives):
    negative_result = []
    for negative in negatives:
        vegetation = Vegetation(
            width, alive_prop=p, control=positive, negative_factor=negative
        )
        vegetation.find_steady_state(1000)
        negative_result.append(vegetation.proportion_alive_list[-1])
    alive_grid.append(negative_result)

alive_grid = np.array(alive_grid)
ax = sns.heatmap(
    alive_grid,
    cmap="viridis",
    xticklabels=[str(x) for x in negatives],
    yticklabels=[str(x) for x in positives],
)
ax.set(xlabel="Negative weight", ylabel="Positive weight")
ax.xaxis.tick_top()
plt.title(f"Proportion of alive cells with starting p={p}")
plt.savefig("appendix_results/proportion_weighted_heatmap.png")
