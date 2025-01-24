import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from tqdm import tqdm

from css_project.fine_grained import FineGrained

p_list = np.logspace(start=-4, stop=0, num=100, base=10)
p_list = np.logspace(start=-2, stop=-1, num=15, base=10)
width = 256

iterations_list = []
alive_list = []

for p in tqdm(p_list):
    model = FineGrained(width, random_seed=None)
    model.initial_grid(p)
    model.find_steady_state(1000)
    iterations = list(range(len(model.proportion_alive_list)))
    alive_list.append(model.proportion_alive_list)
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
plt.legend()
plt.savefig("proportion_alive_over_iter_finegrained.png", dpi=300)

plt.figure(figsize=(8, 6))

steady_alive_list = [x[-1] for x in alive_list]
plt.scatter(p_list, steady_alive_list)

plt.xscale("log")
plt.title("Proportion of Alive Cells vs Initial Density")
plt.xlabel("Initial density")
plt.ylabel("Proportion of Alive Cells")
plt.savefig("proportion_alive_on_probability_finegrained.png", dpi=300)

fig, ax = plt.subplots(layout="constrained")
ax.scatter(p_list, [len(x) for x in iterations_list])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Initial density")
ax.set_ylabel("Iterations till equilibrium")
fig.savefig("iters_till_eq_finegrained.png", dpi=300)
