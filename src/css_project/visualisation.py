import matplotlib.animation as animation
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np

from css_project.fine_grained import FineGrained

from .simple_ca import GameOfLife

QUALITATIVE_COLOURS = [
    "#CCBB44",
    "#228833",
    "#FFFFFF",
    "#4477AA",
    "#AA3377",
    "#66CCEE",
    "#EE6677",
]


def plot_grid(ca: GameOfLife):
    """Plot the static state of a cellular automaton grid.

    Assumes a square grid. Cells are coloured according to a
    qualitative colour pallete.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5), layout="constrained")
    ax.set_axis_off()
    palette = mpc.ListedColormap(
        [QUALITATIVE_COLOURS[i] for i in sorted(np.unique(ca.grid))]
    )
    im_ax = ax.imshow(
        ca.grid,
        cmap=palette,
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

    def update_plot(frame):
        if frame == 0:
            ax.set_data(ca.grid)
        else:
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


def plot_nutrients(ca: FineGrained):
    """Plot the nutrients of a cellular automaton grid.

    Assumes a square grid. Cells are coloured according to nutrient
    level.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5), layout="constrained")
    ax.set_axis_off()
    im_ax = ax.imshow(ca.nutrients, vmin=0.0, vmax=1.0)
    return fig, im_ax


def animate_nutrients(ca: FineGrained, steps: int, fps: int = 5):
    """Creates an animation of a cellular automata nutrients over time.

    Parameterisable via the `steps` (number of updates) and
    `fps` (framerate) parameters.

    Returns a Matplotlib Animation object which must be either
    displayed (plt.show()) or saved (ani.save(FILEPATH)).
    """
    fig, ax = plot_nutrients(ca)

    def update_plot(_):
        ca.update()
        ax.set_data(ca.nutrients)
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
