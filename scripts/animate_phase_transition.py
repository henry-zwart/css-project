"""
Make an animation of the state of a model while the control parameter
is increased and decreased.
"""

import numpy as np

from css_project.vegetation import Vegetation
from css_project.visualisation import animate_phase_transition


def main():
    WIDTH = 128
    control_values = np.linspace(1, 16, 1000)
    control_values = np.concatenate((control_values[::-1], control_values))

    model = Vegetation(WIDTH, alive_prop=0.1, control=16)
    model.run(1000)

    ani = animate_phase_transition(model, control_values, fps=200)
    ani.save("results/phase_transition_2.gif")


if __name__ == "__main__":
    main()
