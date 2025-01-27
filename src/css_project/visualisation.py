import matplotlib.animation as animation
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.pyplot import cm

from css_project.fine_grained import FineGrained

from .vegetation import Vegetation

from .simple_ca import GameOfLife

QUALITATIVE_COLOURS = [
    "#CCBB44",
    "#228833",
    "#AA3377",
    "#FFFFFF",
    "#4477AA",
    "#66CCEE",
    "#EE6677",
]


def plot_grid(ca: GameOfLife, ax: Axes | None = None):
    """Plot the static state of a cellular automaton grid.

    Assumes a square grid. Cells are coloured according to a
    qualitative colour pallete.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5), layout="constrained")
    else:
        fig = None
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

def phase_transition_pos_weight(width, pos_weight_list):
    alive_list = []

    for pos_weight in pos_weight_list:
        vegetation = Vegetation(width, alive_prop=0.5)
        vegetation.positive_factor = pos_weight
        Vegetation.find_steady_state(vegetation, 1000)
        alive_list.append(vegetation.proportion_alive_list)

    steady_alive_list = [x[-1] for x in alive_list]

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(pos_weight_list, steady_alive_list)
    plt.title("Proportion of Alive Cells vs Positive Weight")
    plt.xlabel("Positive Weight")
    plt.ylabel("Proportion of Alive Cells at Steady State")

    return fig
    
def phase_transition_prob(width, p_list, pos_weight_list: int | list[int]):
    # Plot of starting probability vs number of alive cells at equilibrium
    fig2 = plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.title("Proportion of Alive Cells vs Iterations")
    plt.xlabel("Initial Probability")
    plt.ylabel("Proportion of Alive Cells at Steady State")

    if isinstance(pos_weight_list, int):
        pos_weight_list = [pos_weight_list]

    for pos_weight in pos_weight_list: 
        iterations_list = []
        alive_list = []   
        for p in p_list:
            vegetation = Vegetation(width, positive_factor=pos_weight, alive_prop=p)
            vegetation.find_steady_state(1000)
            iterations = list(range(len(vegetation.proportion_alive_list)))
            alive_list.append(vegetation.proportion_alive_list)
            iterations_list.append(iterations)

        steady_alive_list = [x[-1] for x in alive_list]

        plt.scatter(p_list, steady_alive_list, label="Pos. Weight=%d" % (pos_weight))
    plt.legend()

    # Plot of proportion alive cells vs number of iterations for multiple starting probabilities
    fig1 = plt.figure(figsize=(8, 6))

    num_of_lines = len(p_list)
    color = iter(cm.cool(np.linspace(0, 1, num_of_lines)))

    for i in range(len(p_list)):
        c = next(color)
        plt.plot(
            iterations_list[i], alive_list[i], c=c, linestyle="-", label=f"{p_list[i]}"
        )

    plt.title("Proportion of Alive Cells vs Iterations")
    plt.xlabel("Time Step")
    plt.ylabel("Proportion of Alive Cells at Steady State")

    return fig1, fig2