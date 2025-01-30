import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from css_project.vegetation import Vegetation

# from css_project.visualisation import animate_ca, plot_grid
positives = np.linspace(start=1, stop=20, num=20)
initial_probs = np.logspace(start=-3, stop=0, num=100)
width = 64
alive_grid = []

for positive in tqdm(positives):
    prob_result = []
    for prob in initial_probs:
        vegetation = Vegetation(width, alive_prop=prob, control=positive)
        vegetation.find_steady_state(1000)
        prob_result.append(vegetation.proportion_alive_list[-1])
    alive_grid.append(prob_result)

alive_grid = np.array(alive_grid)
ax = sns.heatmap(
    alive_grid,
    cmap="viridis",
    xticklabels=[str(x) for x in initial_probs],
    yticklabels=[str(x) for x in positives],
)
ax.set(xlabel="Initial probability", ylabel="Positive weight")
ax.xaxis.tick_top()
plt.title("Proportion of alive cells")
plt.savefig("proportion_positive_weight_probability_heatmap.png")
