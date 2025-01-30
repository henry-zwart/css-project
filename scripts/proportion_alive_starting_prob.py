import numpy as np

from css_project.visualisation import phase_transition_prob

p_list = np.logspace(start=-4, stop=0, num=100, base=10)
width = 64
pos_weight_list = [5, 7, 9]

fig1, fig2 = phase_transition_prob(width, p_list, pos_weight_list)

fig1.savefig("results/proportion_alive_over_iter", dpi=300)
fig2.savefig("results/proportion_alive_on_probability", dpi=300)
