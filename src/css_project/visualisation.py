import matplotlib.animation as animation
import matplotlib.colors as mpc
import matplotlib.pyplot as plt

from .simple_ca import GameOfLife

QUALITATIVE_COLOURS = [
    "#FFFFFF",
    "#4477AA",
    "#AA3377",
    "#228833",
    "#66CCEE",
    "#CCBB44",
    "#EE6677",
]


def plot_grid(ca: GameOfLife):
    """Plot the static state of a cellular automaton grid.

    Assumes a square grid. Cells are coloured according to a
    qualitative colour pallete.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5), layout="constrained")
    ax.set_axis_off()
    im_ax = ax.imshow(
        ca.grid,
        cmap=mpc.ListedColormap(QUALITATIVE_COLOURS[: ca.N_STATES]),
    )
    return fig, im_ax


def animate_ca(ca: GameOfLife, steps: int, fps: int = 5):
    """Creates an animation of a cellular automata.

    Parameterisable via the `steps` (number of updates) and
    `fps` (framerate) parameters.

    Returns a Matplotlib Animation object which must be either
    displayed (plt.show()) or saved (ani.save(FILEPATH)).
    """
    fig, ax = plot_grid(ca)

    def update_plot(_):
        ca.update()
        ax.set_data(ca.grid)
        return [ax]

    ani = animation.FuncAnimation(
        fig=fig,
        func=update_plot,
        frames=steps,
        interval=1000 / fps,
        repeat=False,
        blit=True,
    )

    return ani
