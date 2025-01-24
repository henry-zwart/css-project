import matplotlib.pyplot as plt
from tqdm import trange

from css_project.complexity import Compression, compressed_size
from css_project.vegetation import Vegetation
from css_project.visualisation import plot_grid


def main():
    # Prepare states
    WIDTH = 64
    p = 0.5
    weight = (3, 4, 6, 8, 10, 13, 17)
    models = [Vegetation(WIDTH, positive_factor=w, alive_prop=p) for w in weight]
    for m in models:
        for _ in trange(50):
            m.update()

    complexities = [[] for _ in Compression]
    for m in models:
        print(f"Positive weight: {m.positive_factor}")
        for i, compression in enumerate(Compression):
            kc = compressed_size(m.grid, compression)
            complexities[i].append(kc)
            print(f"KC({compression}) = {kc:.2f}")
        print()

    fig, ax = plt.subplots(layout="constrained")
    for i, compression in enumerate(Compression):
        ax.plot(weight, complexities[i], label=compression)
        ax.scatter(weight, complexities[i])
    ax.set_xlabel("Positive weight coefficient")
    ax.set_ylabel("Compressed size")
    ax.set_title("KC estimated as compressed size (50 updates)")
    fig.legend()
    fig.savefig("complexity.png")

    for m in models:
        fig, _ = plot_grid(m)
        fig.suptitle(f"Positive weight: {m.positive_factor}")
        fig.savefig(f"complexity_{m.positive_factor}.png")


if __name__ == "__main__":
    main()
