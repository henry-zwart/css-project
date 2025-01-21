from css_project.simple_ca import GameOfLife
from css_project.visualisation import plot_grid

if __name__ == "__main__":
    width = 64

    game_of_life = GameOfLife(width)
    game_of_life.initial_grid(p=0.5)
    fig, ax = plot_grid(game_of_life)
    fig.savefig("test_grid.png", dpi=300)

    t = 0
    total_cells = width * width
    alive = [game_of_life.alive()]

    while t < 10:
        game_of_life.update()
        alive.append(game_of_life.alive())
        t += 1

    fig, ax = plot_grid(game_of_life)
    fig.savefig("test_grid_after.png", dpi=300)

    # Plot ratio of alive vs dead cells
