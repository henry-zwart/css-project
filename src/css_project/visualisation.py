import matplotlib.animation as animation
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.pyplot import cm

from css_project.fine_grained import FineGrained

from .vegetation import Vegetation, InvasiveVegetation
from .logistic import Logistic, LogisticTwoNative
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
    """Creates a plot which calculates the density at equilibrium 
        for a list of positive weights (control). 
    """
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
    """Creates a plot which calculates the density at equilibrium given
        an initial probability to observe a phase transition. 

        This function can make the graph for a list of positive weights 
        to observe the phase transitions for different positive weights.

        Furthermore a plot is created which shows the density of alive cells
        over time given the initial probability. 
    """
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

def densities_invasive_logistic(width, random_seed, p):
    """ Creates a plot for the density of both native and invasive species 
    over time in the logistic model. The invasive species is added
    after a steady state is reached. """
    # Run model until steady state
    model = Logistic(width, random_seed=random_seed)
    model.find_steady_state(1000)

    equilibrium_state = model.grid

    # Introduce invasive species
    invasive_model = LogisticTwoNative(width)
    invasive_model.grid = equilibrium_state
    invasive_model.add_species_to_empty(p=p)

    invasive_model.find_steady_state(1000)

    # Calculate proportions of native, invasive and empty cells
    native = model.proportion_alive_list + invasive_model.proportion_native_alive_list
    invasive = [0] * len(model.proportion_alive_list) + invasive_model.proportion_invasive_alive_list
    iterations = list(range(len(model.proportion_alive_list)+len(invasive_model.proportion_native_alive_list)))

    dead = [1-x for x in [a + b for a, b in zip(native, invasive)]]

    # Plot
    fig1 = plt.figure(figsize=(8, 6))

    plt.plot(iterations, native, label='Native Species')
    plt.plot(iterations, invasive, label='Invasive Species')
    plt.plot(iterations, dead, label='Empty Cells')
    plt.axhline(y = dead[-1], color='r', linestyle='--', linewidth='0.8')
    plt.title("Proportion of Species over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Proportion of Species")
    plt.legend()

    return fig1

def densities_invasive_coarsegrained(width, p_nat, p_inv):
    """ Creates a plot for the density of both native and invasive species 
    over time in the coarse-grained model. The invasive species is added
    after a steady state is reached. """
    # Run model until steady state
    model = InvasiveVegetation(width, species_prop=(p_nat, 0))
    model.run(100)
   
    
    model.introduce_invasive(p_inv=p_inv)

    model.run(100)

    # Calculate proportions of native, invasive and empty cells
    native = model.proportion_native_alive_list
    invasive = model.proportion_invasive_alive_list
    iterations = list(range(len(model.proportion_native_alive_list)))

    dead = [1-x for x in [a + b for a, b in zip(native, invasive)]]

    # Plot
    fig1 = plt.figure(figsize=(8, 6))

    plt.plot(iterations, native, label='Native Species')
    plt.plot(iterations, invasive, label='Invasive Species')
    plt.plot(iterations, dead, label='Empty Cells')
    plt.axhline(y = dead[-1], color='r', linestyle='--', linewidth='0.8')
    plt.title("Proportion of Species over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Proportion of Species")
    plt.legend()

    return fig1