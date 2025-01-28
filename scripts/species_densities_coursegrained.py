import matplotlib.pyplot as plt
import numpy as np

from css_project.vegetation import Vegetation
from css_project.visualisation import densities_invasive_coursegrained

width = 128
p_inv = 0.3

fig1 = densities_invasive_coursegrained(width, p_inv)

fig1.savefig("species_densities_coursegrained", dpi=300)