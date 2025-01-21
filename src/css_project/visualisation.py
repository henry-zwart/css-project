import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import seaborn as sns

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
    n_states = ca.grid.max() + 1
    fig, ax = plt.subplots(figsize=(6.5, 6.5), layout="constrained")
    ax.set_axis_off()
    sns.heatmap(
        ca.grid,
        cbar=False,
        square=True,
        cmap=mpc.ListedColormap(QUALITATIVE_COLOURS[:n_states]),
        ax=ax,
    )
    return fig, ax
