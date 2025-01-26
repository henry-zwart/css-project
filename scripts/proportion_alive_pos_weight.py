import matplotlib.pyplot as plt
import numpy as np

from css_project.vegetation import Vegetation

# from css_project.visualisation import animate_ca, plot_grid
pos_weight_list = np.linspace(0, 20, 21)

width = 64

iterations_list = []
alive_list = []

for pos_weight in pos_weight_list:
    vegetation = Vegetation(width, alive_prop=0.5)
    vegetation.positive_factor = pos_weight
    Vegetation.find_steady_state(vegetation, 1000)
    alive_list.append(vegetation.proportion_alive_list)

plt.figure(figsize=(8, 6))

steady_alive_list = [x[-1] for x in alive_list]
plt.scatter(pos_weight_list, steady_alive_list)

plt.title("Proportion of Alive Cells vs Positive Weight")
plt.xlabel("Positive Weight")
plt.ylabel("Proportion of Alive Cells")
plt.savefig("proportion_alive_on_pos_weight", dpi=300)
