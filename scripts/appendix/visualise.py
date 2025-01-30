import matplotlib.pyplot as plt

from css_project.simple_ca import GameOfLife
from css_project.visualisation import animate_ca, plot_grid

if __name__ == "__main__":
    width = 128

    game_of_life = GameOfLife(width)
    game_of_life.initial_grid(p=0.5)

    # Copy grid to use in graph
    initial_grid = game_of_life.grid.copy()

    ani = animate_ca(game_of_life, 1000)
    ani.save("game_of_life.gif")

    # Reset grid to initial state
    game_of_life.grid = initial_grid.copy()

    fig, ax = plot_grid(game_of_life)
    fig.savefig("appendix_results/test_grid.png", dpi=300)

    # Plot the proportion of alive cells over time
    t = 0
    total_cells = width * width
    alive = [game_of_life.total_alive() / total_cells]

    while t < 1000:
        game_of_life.update()
        alive.append(game_of_life.total_alive() / total_cells)
        t += 1
    print(alive)

    iterations = list(range(len(alive)))

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, alive, linestyle="-")
    plt.title("Proportion of Alive Cells vs Iterations")
    plt.xlabel("Time Step")
    plt.ylabel("Proportion of Alive Cells")
    plt.savefig("appendix_results/proportion_alive.png", dpi=300)
