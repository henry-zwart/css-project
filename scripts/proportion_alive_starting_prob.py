import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from tqdm import tqdm

from css_project.vegetation import Vegetation

# from css_project.visualisation import animate_ca, plot_grid
p_list = [
    0.001,
    0.0015,
    0.0018,
    0.002,
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.01,
    0.1,
]

p_list = np.logspace(start=-4, stop=0, num=100, base=10)
width = 64

iterations_list = []
alive_list = []

for p in tqdm(p_list):
    vegetation = Vegetation(width, alive_prop=p)
    vegetation.find_steady_state(1000)
    iterations = list(range(len(vegetation.proportion_alive_list)))
    alive_list.append(vegetation.proportion_alive_list)
    iterations_list.append(iterations)


plt.figure(figsize=(8, 6))

num_of_lines = len(p_list)
color = iter(cm.cool(np.linspace(0, 1, num_of_lines)))

for i in range(len(p_list)):
    c = next(color)
    plt.plot(
        iterations_list[i], alive_list[i], c=c, linestyle="-", label=f"{p_list[i]}"
    )


plt.title("Proportion of Alive Cells vs Iterations")
plt.xlabel("Time Step")
plt.ylabel("Proportion of Alive Cells")
plt.savefig("proportion_alive_over_iter", dpi=300)

plt.figure(figsize=(8, 6))

steady_alive_list = [x[-1] for x in alive_list]
plt.scatter(p_list, steady_alive_list)

plt.xscale("log")
plt.title("Proportion of Alive Cells vs Iterations")
plt.xlabel("Time Step")
plt.ylabel("Proportion of Alive Cells")
plt.savefig("proportion_alive_on_probability", dpi=300)
