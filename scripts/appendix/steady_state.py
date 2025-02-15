"""Find the steady state and visualize the proportion of alive cells at the moment"""

import matplotlib.pyplot as plt

from css_project.vegetation import Vegetation

# from css_project.visualisation import animate_ca, plot_grid


width = 64
vegetation = Vegetation(width, alive_prop=0.3)
t = 0
total_cells = width * width

Vegetation.find_steady_state(vegetation, 200)

iterations = list(range(len(vegetation.proportion_alive_list)))

plt.figure(figsize=(8, 6))
plt.plot(iterations, vegetation.proportion_alive_list, linestyle="-")
plt.title("Proportion of Alive Cells vs Iterations")
plt.xlabel("Time Step")
plt.ylabel("Proportion of Alive Cells")
plt.savefig("appendix_results/steady_alive_vegetation.png", dpi=300)
