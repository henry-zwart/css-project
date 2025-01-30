"""Find the steady state and visualize the proportion of alive cells at the moment"""

import matplotlib.pyplot as plt

from css_project.logistic import Logistic

width = 256
model = Logistic(
    width,
    consume_rate=63.7,
    control=35,
    alive_prop=0.01,
)
t = 0
total_cells = width * width

model.find_steady_state(1000)

iterations = list(range(len(model.proportion_alive_list)))

plt.figure(figsize=(8, 6))
plt.plot(iterations, model.proportion_alive_list, linestyle="-")
plt.title("Proportion of Alive Cells vs Iterations")
plt.xlabel("Time Step")
plt.ylabel("Proportion of Alive Cells")
plt.savefig("appendix_results/steady_alive_finegrained.png", dpi=300)
