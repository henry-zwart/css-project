import numpy as np

from css_project.visualisation import phase_transition_pos_weight

pos_weight_list = np.linspace(0, 20, 21)
width = 64

fig = phase_transition_pos_weight(width, pos_weight_list)
fig.savefig("appendix_results/proportion_alive_on_pos_weight", dpi=300)
