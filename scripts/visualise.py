from css_project.simple_ca import GameOfLife
from css_project.visualisation import animate_ca, plot_grid

if __name__ == "__main__":
    width = 64

    game_of_life = GameOfLife(width)
    game_of_life.initial_grid(p=0.5)
    fig, ax = plot_grid(game_of_life)
    fig.savefig("test_grid.png", dpi=300)

    t = 0
    total_cells = width * width
    alive = [game_of_life.total_alive()]

    while t < 10:
        game_of_life.update()
        alive.append(game_of_life.total_alive())
        t += 1

    fig, ax = plot_grid(game_of_life)
    fig.savefig("test_grid_after.png", dpi=300)

    # Plot ratio of alive vs dead cells
    game_of_life = GameOfLife(128)
    game_of_life.initial_grid(p=0.5)
    ani = animate_ca(game_of_life, 30)
    ani.save("game_of_life.mp4")
