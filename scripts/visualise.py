from css_project.simple_ca import GameOfLife
from css_project.visualisation import plot_grid

if __name__ == "__main__":
    game_of_life = GameOfLife(64)
    game_of_life.initial_grid(p=0.5)
    fig, ax = plot_grid(game_of_life)
    fig.savefig("test_grid.png", dpi=300)
