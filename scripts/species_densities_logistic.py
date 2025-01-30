from css_project.visualisation import densities_invasive_logistic

width = 128
random_seed = 1
p = 0.9

fig1 = densities_invasive_logistic(width, random_seed, p)

fig1.savefig("species_densities_logistic", dpi=300)
