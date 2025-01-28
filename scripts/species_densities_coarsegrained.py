import matplotlib.pyplot as plt
import numpy as np

from css_project.vegetation import Vegetation
from css_project.visualisation import densities_invasive_coarsegrained

width = 128
p_nat = 0.25
p_inv = 0.3

fig1 = densities_invasive_coarsegrained(width, p_nat, p_inv)

fig1.savefig("species_densities_coarsegrained", dpi=300)