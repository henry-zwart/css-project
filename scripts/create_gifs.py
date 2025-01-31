"""Make gifs of the coarse-grained model and the logistic model without
invasive species for three values of control parameters."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, sum

from css_project.vegetation import Vegetation
from css_project.logistic import LogisticTwoNative
from css_project.visualisation import animate_ca

if __name__ == "__main__":
    width = 64
    pos_list = [6, 9, 13]

    # This code makes gifs of activator-inhibitor model for multiple control parameters
    for pos in pos_list:
        vegetation = Vegetation(width, control=pos)
        vegetation.initial_grid(p=0.3)

        ani = animate_ca(vegetation, 60)
        ani.save(f"results/activator_inhibitor_control={pos}.gif")

    control_list = [10, 30, 50]
    for control_val in control_list:
        model = LogisticTwoNative(
            width,
            consume_rate_1=63.7,
            control=control_val,
            species_prop=[0.001, 0.0],
        )
        ani = animate_ca(model, 1000, fps=30)
        ani.save(f"results/logistic_control={control_val}.gif")