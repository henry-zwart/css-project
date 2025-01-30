"""Find the steady state and visualize the proportion of alive cells at the moment"""

import matplotlib.pyplot as plt

from css_project.fine_grained import FineGrained

width = 256
model = FineGrained(
    width,
    nutrient_level=1.0,
    nutrient_diffusion_rate=0.21,
    nutrient_consume_rate=0.1,
    nutrient_regenerate_rate=0.8,
)
model.initial_grid(0.05)
t = 0
total_cells = width * width

model.find_steady_state(1000)

iterations = list(range(len(model.proportion_alive_list)))

plt.figure(figsize=(8, 6))
plt.plot(iterations, model.proportion_alive_list, linestyle="-")
plt.title("Proportion of Alive Cells vs Iterations")
plt.xlabel("Time Step")
plt.ylabel("Proportion of Alive Cells")
plt.savefig("steady_alive_finegrained.png", dpi=300)
