import matplotlib.pyplot as plt

from css_project.vegetation import Vegetation

# from css_project.visualisation import animate_ca, plot_grid

width = 64
vegetation = Vegetation(width)
vegetation.initial_grid(0.3)
t = 0
total_cells = width * width
alive = [vegetation.total_alive() / total_cells]
while t < 300:
    vegetation.update()
    alive.append(vegetation.total_alive() / total_cells)
    t += 1
print(alive)

iterations = list(range(len(alive)))

plt.figure(figsize=(8, 6))
plt.plot(iterations, alive, linestyle="-")
plt.title("Proportion of Alive Cells vs Iterations")
plt.xlabel("Time Step")
plt.ylabel("Proportion of Alive Cells")
plt.savefig("proportion_alive_vegetation.png", dpi=300)
