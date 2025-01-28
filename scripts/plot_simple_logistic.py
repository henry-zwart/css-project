from tqdm import trange

from css_project.logistic import LogisticTwoNative
from css_project.visualisation import animate_ca

if __name__ == "__main__":
    width = 256

    model = LogisticTwoNative(
        width,
        consume_rate_1=63.7,
        consume_rate_2=63.7 * (0.5),
        control=35,
        species_prop=[0.001, 0.0],
    )
    for _ in trange(500):
        model.update()

    model.add_species_to_empty(1.0)
    ani = animate_ca(model, 1000, fps=30)
    ani.save("logistic_two_species.mp4")
