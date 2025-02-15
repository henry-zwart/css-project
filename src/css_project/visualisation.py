"""
This Module Contains several functions to visualise graphs and animations
for different models.
"""

from collections.abc import Sequence

import matplotlib.animation as animation
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
from tqdm import trange

from css_project.complexity import (
    count_clusters,
    fluctuation_cluster_size,
    maximum_cluster_size,
    ratio_cluster_size,
    variance_cluster_size,
)
from css_project.fine_grained import FineGrained
from css_project.logistic import Logistic
from css_project.model import VegetationModel

from .logistic import LogisticTwoNative
from .vegetation import InvasiveVegetation, Vegetation

QUALITATIVE_COLOURS = [
    "#CCBB44",
    "#228833",
    "#AA3377",
    "#4477AA",
    "#66CCEE",
    "#EE6677",
]


def confidence(arr: np.ndarray, z: float = 1.97) -> np.ndarray:
    """Compute confidence interval size.

    Uses sample standard deviation.

    Args:
        arr: Numpy array with first dimension as repeats
        z: Z-score to use for confidence interval

    Returns:
        Confidence interval width. Distance from mean value.
    """
    n_repeats = arr.shape[0]
    return z * arr.std(ddof=1, axis=0) / np.sqrt(n_repeats)


def mean_and_cis(
    arr: np.ndarray, z: float = 1.97, rm_nan: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and lower/upper confidence intervals for an array.

    Uses sample standard deviation to compute confidence intervals.

    Args:
        arr: Numpy array with repeats in first dimension
        z: Z-score to use for confidence intervals

    Returns:
        Tuple of three arrays (mean, lower ci, upper ci) with the same
        shape as input array, excluding the repeat dimension.
    """
    mean = np.nanmean(arr, axis=0) if rm_nan else arr.mean(axis=0)
    conf_interval = confidence(arr, z)
    return mean, mean - conf_interval, mean + conf_interval


def plot_grid(model: VegetationModel, ax: Axes | None = None):
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
        [QUALITATIVE_COLOURS[i] for i in sorted(np.unique(model.grid))]
    )
    im_ax = ax.imshow(
        model.grid,
        cmap=palette,
    )
    return fig, im_ax


def animate_ca(model: VegetationModel, steps: int, fps: int = 5):
    """Creates an animation of a cellular automata.

    Parameterisable via the `steps` (number of updates) and
    `fps` (framerate) parameters.

    Returns a Matplotlib Animation object which must be either
    displayed (plt.show()) or saved (ani.save(FILEPATH)).
    """
    fig, ax = plot_grid(model)

    def update_plot(frame: int):
        """Updates the model plot for frames higher than zero."""
        if frame == 0:
            ax.set_data(model.grid)
        else:
            model.update()
            ax.set_data(model.grid)
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
        """Updates the corresponding model plot."""
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


def phase_transition_pos_weight(width: int, pos_weight_list: list):
    """Creates a plot which calculates the density at equilibrium
    for a list of positive weights (control) and returns a figure.
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


def phase_transition_prob(width: int, p_list, pos_weight_list: int | list[int]):
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
            vegetation = Vegetation(width, control=pos_weight, alive_prop=p)
            vegetation.find_steady_state(1000)
            iterations = list(range(len(vegetation.proportion_alive_list)))
            alive_list.append(vegetation.proportion_alive_list)
            iterations_list.append(iterations)

        steady_alive_list = [x[-1] for x in alive_list]

        plt.scatter(p_list, steady_alive_list, label=f"Pos. Weight={pos_weight}")
    plt.legend()

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


def densities_invasive_logistic(width: int, random_seed, p):
    """Creates a plot for the density of both native and invasive species
    over time in the logistic model. The invasive species is added
    after a steady state is reached."""
    # Run model until steady state
    model = Logistic(width, random_seed=random_seed)
    model.find_steady_state(1000)

    equilibrium_state = model.grid

    # Introduce invasive species
    invasive_model = LogisticTwoNative(width)
    invasive_model.grid = equilibrium_state
    invasive_model.introduce_invasive(p_inv=p)

    invasive_model.find_steady_state(1000)

    # Calculate proportions of native, invasive and empty cells
    native = model.proportion_alive_list + invasive_model.proportion_native_alive_list
    invasive = [0] * len(
        model.proportion_alive_list
    ) + invasive_model.proportion_invasive_alive_list
    iterations = list(
        range(
            len(model.proportion_alive_list)
            + len(invasive_model.proportion_native_alive_list)
        )
    )

    dead = [1 - x for x in [a + b for a, b in zip(native, invasive, strict=False)]]

    # Plot
    fig1 = plt.figure(figsize=(8, 6))

    plt.plot(iterations, native, label="Native Species")
    plt.plot(iterations, invasive, label="Invasive Species")
    plt.plot(iterations, dead, label="Empty Cells")
    plt.axhline(y=dead[-1], color="r", linestyle="--", linewidth="0.8")
    plt.title("Proportion of Species over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Proportion of Species")
    plt.legend()

    return fig1


def densities_invasive_coarsegrained(width: int, p_nat: float, p_inv: float):
    """Creates a plot for the density of both native and invasive species
    over time in the activator-inhibitor model. The invasive species is added
    after a steady state is reached."""
    # Run model until steady state
    native = []
    invasive = []
    dead = []
    total = width * width

    model = InvasiveVegetation(width, species_prop=(p_nat, 0))
    native.append((model.species_alive()[0]) / total)
    invasive.append((model.species_alive()[1]) / total)
    dead.append((total - (model.species_alive()[0] + model.species_alive()[1])) / total)

    t = 0
    timespan = 25
    while t < timespan:
        model.update()
        native.append((model.species_alive()[0]) / total)
        invasive.append((model.species_alive()[1]) / total)
        dead.append(
            (total - (model.species_alive()[0] + model.species_alive()[1])) / total
        )
        t += 1

    model.introduce_invasive(p_inv=p_inv)
    native.append((model.species_alive()[0]) / total)
    invasive.append((model.species_alive()[1]) / total)
    dead.append((total - (model.species_alive()[0] + model.species_alive()[1])) / total)

    t = 0
    timespan = 25
    while t < timespan:
        model.update()
        native.append((model.species_alive()[0]) / total)
        invasive.append((model.species_alive()[1]) / total)
        dead.append(
            (total - (model.species_alive()[0] + model.species_alive()[1])) / total
        )
        t += 1

    iterations = list(range(len(native)))

    # Plot
    fig1 = plt.figure(figsize=(8, 6))

    plt.plot(iterations, native, label="Native Species")
    plt.plot(iterations, invasive, label="Invasive Species")
    plt.plot(iterations, dead, label="Empty Cells")
    plt.axhline(y=native[-1], color="r", linestyle="--", linewidth="0.8")
    plt.title("Proportion of Species over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Proportion of Species")
    plt.legend()

    return fig1


def distribution_leftright_steps(
    n_steps: int,
    min_val: float,
    init_val: float,
    max_val: float,
) -> tuple[int, int]:
    """
    Given an initial value in an interval and a total number of steps
    calculate the amount of steps to the left and to the right that are needed
    to move from the inital value.

    Args:
        n_steps: Number of total steps to divide the interval
        min_val: Minimal value of interval
        init_val: Starting value
        max_val: Maximal value of interval

    Returns:
        Tuple of the amount of steps to the left boundary of the interval
        starting from the initial value and the amount of steps to the
        right boundary.
    """
    right_range = max_val - init_val
    total_range = max_val - min_val
    right_proportion = right_range / total_range
    right_steps = int(right_proportion * n_steps)
    left_steps = n_steps - right_steps
    return (left_steps, right_steps)


def logistic_phase_plot(
    model_type,
    widths: list[int],
    init_density: float,
    init_controls: list[float],
    min_nutrient: float,
    max_nutrient: float,
    n_steps: int,
    iters_per_step: int,
    n_repeats: int = 1,
    control_variable_name: str = "Control variable",
) -> tuple[Figure, Axes]:
    """
    Visualise phase transitions in both a coarse-grained model and logistic model
    from an initial control parameter. The model is run until a steady state
    before varying the control parameter. The used
    order parameters are density at equilibrium, cluster ratio, maximum
    cluster size and cluster fluctuation.

    Assume that only native species are involved.
    """
    fig, axes = plt.subplots(
        nrows=3, ncols=1, layout="constrained", figsize=(8, 6), sharex=True
    )

    colors = ["#1f77b4", "#ff7f0e"]
    linestyles = ["dashed", "dashdot", "solid"]

    for width_i, width in enumerate(widths):
        for init_i, init_nutrient in enumerate(init_controls):
            # Distribute n_steps between left and right
            left_steps, right_steps = distribution_leftright_steps(
                n_steps,
                min_nutrient,
                init_nutrient,
                max_nutrient,
            )

            left_nutrients = np.linspace(min_nutrient, init_nutrient, left_steps)
            right_nutrients = np.linspace(init_nutrient, max_nutrient, right_steps)
            nutrient_vals = np.concat((left_nutrients[:-1], right_nutrients))

            equilibrium_density = np.zeros((n_repeats, n_steps - 1), dtype=np.float64)
            equilibrium_n_clusters = np.zeros_like(equilibrium_density)
            equilibrium_fluctuation = np.zeros_like(equilibrium_density)

            for repeat in trange(n_repeats):
                # Run model till equilibrium
                model = model_type(
                    width,
                    control=init_nutrient,
                    alive_prop=init_density,
                    random_seed=repeat,
                )
                model.run(1000)
                initial_grid = model.grid.copy()

                equilibrium_density[repeat, left_steps - 1] = model.proportion_alive()
                equilibrium_n_clusters[repeat, left_steps - 1] = (
                    count_clusters(model.grid) / model.area
                )
                equilibrium_fluctuation[repeat, left_steps - 1] = (
                    fluctuation_cluster_size(model.grid)
                )

                # Vary control parameter upward
                for i, ns in enumerate(nutrient_vals[left_steps:], start=left_steps):
                    model.set_control(ns)
                    model.run(iters_per_step)
                    equilibrium_density[repeat, i] = model.proportion_alive()
                    equilibrium_n_clusters[repeat, i] = (
                        count_clusters(model.grid) / model.area
                    )
                    equilibrium_fluctuation[repeat, i] = fluctuation_cluster_size(
                        model.grid
                    )

                # Reset grid and vary control parameter downward
                model.grid = initial_grid
                for bw_steps, ns in enumerate(
                    nutrient_vals[left_steps - 2 :: -1], start=2
                ):
                    i = left_steps - bw_steps
                    model.set_control(ns)
                    model.run(iters_per_step)
                    equilibrium_density[repeat, i] = model.proportion_alive()
                    equilibrium_n_clusters[repeat, i] = (
                        count_clusters(model.grid) / model.area
                    )
                    #   equilibrium_max_cluster[repeat, i] = maximum_cluster_size(
                    #       model.grid
                    #   )
                    equilibrium_fluctuation[repeat, i] = fluctuation_cluster_size(
                        model.grid
                    )

            density_mean, density_lower, density_upper = mean_and_cis(
                equilibrium_density
            )
            nc_mean, nc_lower, nc_upper = mean_and_cis(equilibrium_n_clusters)
            fluc_mean, fluc_lower, fluc_upper = mean_and_cis(
                equilibrium_fluctuation, rm_nan=True
            )

            color = colors[init_i]
            linestyle = linestyles[width_i]
            # Plot density
            axes[0].plot(
                nutrient_vals,
                density_mean,
                linestyle=linestyle,
                color=color,
                label=f"{init_nutrient:.2f}",
            )
            axes[0].fill_between(
                nutrient_vals, density_lower, density_upper, color=color, alpha=0.3
            )
            axes[0].set_ylabel("Equilibrium density")
            axes[0].set_ylim(0, 1)
            axes[0].axvline(init_nutrient, linestyle="dashed", color=color)

            # Plot number of clusters
            axes[1].plot(
                nutrient_vals,
                nc_mean,
                linestyle=linestyle,
                color=color,
                label=f"{init_nutrient:.2f}",
            )
            axes[1].fill_between(
                nutrient_vals, nc_lower, nc_upper, color=color, alpha=0.3
            )
            axes[1].set_ylabel("Cluster count")
            axes[1].axvline(init_nutrient, linestyle="dashed", color=color)

            # Plot number of clusters
            axes[2].plot(
                nutrient_vals,
                fluc_mean,
                linestyle=linestyle,
                color=color,
                label=f"{init_nutrient:.2f}",
            )
            axes[2].fill_between(
                nutrient_vals, fluc_lower, fluc_upper, color=color, alpha=0.3
            )
            axes[2].set_ylabel("Cluster fluctuation")
            axes[2].set_yscale("log")
            axes[2].axvline(init_nutrient, linestyle="dashed", color=color)

    linestyle_handles = [
        plt.Line2D([0], [0], color="grey", linestyle=linestyles[i], label=widths[i])
        for i in range(len(widths))
    ]

    fig.legend(handles=linestyle_handles)
    fig.supxlabel(control_variable_name)

    return fig, axes


def invasive_phase_plot(
    model_type,
    width: int,
    init_density: float,
    init_nutrient: float,
    min_nutrient: float,
    max_nutrient: float,
    p_inv: float,
    n_steps: int,
    iters_per_step: int,
    n_repeats: int = 1,
) -> tuple[Figure, Axes]:
    """
    Visualise phase transitions in both a coarse-grained model and logistic model
    from an initial control parameter. The model is run until a steady state when
    invasive species are introduced. Then run the model again before varying the
    control parameter. The used order parameters are density at equilibrium,
    cluster ratio, maximum cluster size and cluster fluctuation.
    """
    fig, axes = plt.subplots(
        nrows=6, ncols=1, layout="constrained", figsize=(8, 10), sharex=True
    )

    # Distribute n_steps between left and right
    left_steps, right_steps = distribution_leftright_steps(
        n_steps,
        min_nutrient,
        init_nutrient,
        max_nutrient,
    )

    left_nutrients = np.linspace(min_nutrient, init_nutrient, left_steps)
    right_nutrients = np.linspace(init_nutrient, max_nutrient, right_steps)
    nutrient_vals = np.concat((left_nutrients[:-1], right_nutrients))

    equilibrium_density = np.zeros((n_repeats, n_steps - 1), dtype=np.float64)
    equilibrium_cluster_ratio = np.zeros_like(equilibrium_density)
    equilibrium_n_clusters = np.zeros_like(equilibrium_density)
    equilibrium_max_cluster = np.zeros_like(equilibrium_density)
    equilibrium_variance = np.zeros_like(equilibrium_density)
    equilibrium_fluctuation = np.zeros_like(equilibrium_density)

    for repeat in trange(n_repeats):
        # Run model till equilibrium, introduce invasive, run model again
        model = model_type(
            width,
            control=init_nutrient,
            species_prop=(init_density, 0),
        )
        model.run(1000)
        model.introduce_invasive(p_inv)
        model.run(1000)

        initial_grid = model.grid.copy()

        equilibrium_density[repeat, left_steps - 1] = (
            model.species_alive()[0] / model.area
        )
        equilibrium_cluster_ratio[repeat, left_steps - 1] = ratio_cluster_size(
            model.grid
        )
        equilibrium_n_clusters[repeat, left_steps - 1] = count_clusters(model.grid)
        equilibrium_max_cluster[repeat, left_steps - 1] = maximum_cluster_size(
            model.grid
        )
        equilibrium_variance[repeat, left_steps - 1] = variance_cluster_size(model.grid)
        equilibrium_fluctuation[repeat, left_steps - 1] = fluctuation_cluster_size(
            model.grid
        )

        # Vary control parameter upward
        for i, ns in enumerate(nutrient_vals[left_steps:], start=left_steps):
            model.set_control(ns)
            model.run(iters_per_step)
            equilibrium_density[repeat, i] = model.species_alive()[0] / model.area
            equilibrium_cluster_ratio[repeat, i] = ratio_cluster_size(model.grid)
            equilibrium_n_clusters[repeat, i] = count_clusters(model.grid)
            equilibrium_max_cluster[repeat, i] = maximum_cluster_size(model.grid)
            equilibrium_variance[repeat, i] = variance_cluster_size(model.grid)
            equilibrium_fluctuation[repeat, i] = fluctuation_cluster_size(model.grid)

        # Reset grid and vary control parameter downward
        model.grid = initial_grid
        for bw_steps, ns in enumerate(nutrient_vals[left_steps - 2 :: -1], start=2):
            i = left_steps - bw_steps
            model.set_control(ns)
            model.run(iters_per_step)
            equilibrium_density[repeat, i] = model.species_alive()[0] / model.area
            equilibrium_cluster_ratio[repeat, i] = ratio_cluster_size(model.grid)
            equilibrium_n_clusters[repeat, i] = count_clusters(model.grid)
            equilibrium_max_cluster[repeat, i] = maximum_cluster_size(model.grid)
            equilibrium_variance[repeat, i] = variance_cluster_size(model.grid)
            equilibrium_fluctuation[repeat, i] = fluctuation_cluster_size(model.grid)

    axes[0].plot(nutrient_vals, equilibrium_density.min(axis=0))
    axes[0].plot(nutrient_vals, equilibrium_density.max(axis=0))
    axes[0].set_ylabel("Equilibrium density")
    axes[0].set_ylim(0, 1)

    axes[1].plot(nutrient_vals, equilibrium_n_clusters.mean(axis=0))
    axes[1].vlines(
        init_nutrient, ymin=0, ymax=equilibrium_n_clusters.mean(axis=0)[left_steps - 1]
    )
    axes[1].set_ylabel("Cluster count")
    axes[1].set_ylim(0, None)

    axes[2].plot(nutrient_vals, equilibrium_cluster_ratio.mean(axis=0))
    axes[2].vlines(
        init_nutrient,
        ymin=0,
        ymax=equilibrium_cluster_ratio.mean(axis=0)[left_steps - 1],
    )
    axes[2].set_ylabel("Cluster size ratio")
    axes[2].set_ylim(0, None)

    axes[3].plot(nutrient_vals, equilibrium_max_cluster.mean(axis=0))
    axes[3].vlines(
        init_nutrient, ymin=0, ymax=equilibrium_max_cluster.mean(axis=0)[left_steps - 1]
    )
    axes[3].set_ylabel("Giant component")
    axes[3].set_yscale("log")

    axes[4].plot(nutrient_vals, equilibrium_variance.mean(axis=0))
    axes[4].vlines(
        init_nutrient,
        ymin=0,
        ymax=equilibrium_variance.mean(axis=0)[left_steps - 1],
    )
    axes[4].set_ylabel("Cluster size variance")
    axes[4].set_yscale("log")

    axes[5].plot(nutrient_vals, equilibrium_fluctuation.mean(axis=0))
    axes[5].vlines(
        init_nutrient,
        ymin=0,
        ymax=equilibrium_fluctuation.mean(axis=0)[left_steps - 1],
    )
    axes[5].set_ylabel("Cluster size fluctuation")
    axes[5].set_yscale("log")

    fig.supxlabel("Nutrient supplementation")

    return fig, axes


def animate_phase_transition(
    model: VegetationModel, control_values: Sequence[float], fps: int = 60
):
    """Given a model and a list of control values return an animation of
    the states while the control value is being increased and decreased.

    Returns a Matplotlib Animation object which must be either
    displayed (plt.show()) or saved (ani.save(FILEPATH)).
    """
    fig, ax = plot_grid(model)

    def update_plot(frame):
        model.set_control(control_values[frame])
        model.update()
        ax.set_data(model.grid)
        return [ax]

    ani = animation.FuncAnimation(
        fig=fig,
        func=update_plot,
        frames=len(control_values),
        interval=(1000 / fps),
        repeat=False,
        blit=True,
    )
    return ani


def count_states(alive_nat: list, alive_inv: list, vegetation, total_cells: int):
    """Counts the number of cells in the native and invasive states.
    Adds the number to a list for the respective states and returns
    the list."""
    alive_n, alive_i = vegetation.species_alive()
    alive_nat.append(alive_n / total_cells)
    alive_inv.append(alive_i / total_cells)

    return alive_nat, alive_inv


def plot_invasive_model_multiple(
    width: int,
    p_nat: float,
    pp_inv: float,
    timespan: int = 40,
    initial_state: str = "equilibrium",
    t_eq: int = 30,
    inv_type: str = "empty",
):
    """This Code runs the invasive model multiple times to plot the
    proportion of native species over time for different introduced
    proportions of invasive.

    Args:
        width: The number of cells in one side of the grid.
        p_nat: Initial proportion of native species.
        p_inv: Initial proportion of invasive species.
        timespan: Number of iterations/amount of time to obtain data.
        initial_state: The specific state in which the native species
            finds itself.
        t_eq: Time to run the updates until equilibrium.
        inv_type: How we introduce invasive species. Assumes only one
            empty cells."""
    if initial_state == "equilibrium":
        vegetation = InvasiveVegetation(width, species_prop=(p_nat, 0.0))

        t = 0
        total_cells = vegetation.width * vegetation.width

        while t < t_eq:
            vegetation.update()
            t += 1

        # Copy grid to save initial state before invasive species introduction
        initial_grid = vegetation.grid.copy()

        alive_nat_tot = []
        alive_inv_tot = []
        alive_nat = []
        alive_inv = []

        for p_inv in pp_inv:
            # Introduce invasive species
            alive_nat, alive_inv = count_states(
                alive_nat, alive_inv, vegetation, total_cells
            )
            vegetation.introduce_invasive(p_inv, inv_type)
            alive_nat, alive_inv = count_states(
                alive_nat, alive_inv, vegetation, total_cells
            )

            t = 0
            while t < timespan:
                vegetation.update()
                alive_nat, alive_inv = count_states(
                    alive_nat, alive_inv, vegetation, total_cells
                )
                t += 1

            # Save proportions for run
            alive_nat_tot.append(alive_nat[:])
            alive_inv_tot.append(alive_inv[:])

            # Reset
            vegetation.grid = initial_grid.copy()
            t = 0
            alive_nat = []
            alive_inv = []

        # Plotting
        plt.figure(figsize=(10, 8))
        colors = cm.viridis(np.linspace(0, 1, len(pp_inv)))

        for i, (p_inv, color) in enumerate(zip(pp_inv, colors, strict=False)):
            iterations = list(range(len(alive_nat_tot[i])))
            plt.plot(
                iterations,
                alive_nat_tot[i],
                color=color,
                label=f"Native (p_inv={p_inv})",
                linestyle="-",
            )

        plt.title("Proportion of Native Species for different p_inv")
        plt.xlabel("Time Step")
        plt.ylabel("Proportion of Cells")
        plt.legend(loc="upper right", fontsize="small")
        plt.savefig(f"results/proportion_nat_gradient_eq_p_nat={p_nat}.png", dpi=300)
        plt.show()

    elif initial_state == "random":
        vegetation = InvasiveVegetation(width, species_prop=(0.25, 0.25))
        t = 0
        total_cells = vegetation.width * vegetation.width

        # Copy grid to save initial state
        initial_grid = vegetation.grid.copy()

        alive_nat_tot = []
        alive_inv_tot = []
        alive_nat = []
        alive_inv = []

        for p_inv in pp_inv:
            # Introduce invasive species
            alive_nat, alive_inv = count_states(
                alive_nat, alive_inv, vegetation, total_cells
            )
            vegetation.introduce_invasive(p_inv, inv_type)
            alive_nat, alive_inv = count_states(
                alive_nat, alive_inv, vegetation, total_cells
            )

            t = 0
            while t < timespan:
                vegetation.update()
                alive_nat, alive_inv = count_states(
                    alive_nat, alive_inv, vegetation, total_cells
                )
                t += 1

            # Save proportions for run
            alive_nat_tot.append(alive_nat[:])
            alive_inv_tot.append(alive_inv[:])

            # Reset
            vegetation.grid = initial_grid.copy()
            t = 0
            alive_nat = []
            alive_inv = []

        # Plotting
        plt.figure(figsize=(10, 8))
        colors = cm.viridis(np.linspace(0, 1, len(pp_inv)))

        for i, (p_inv, color) in enumerate(zip(pp_inv, colors, strict=False)):
            iterations = list(range(len(alive_nat_tot[i])))
            plt.plot(
                iterations,
                alive_nat_tot[i],
                color=color,
                label=f"Native (p_inv={p_inv})",
                linestyle="-",
            )

        plt.title("Proportion of Native Species for different p_inv")
        plt.xlabel("Time Step")
        plt.ylabel("Proportion of Cells")
        plt.legend(loc="upper right", fontsize="small")
        plt.savefig(
            f"results/proportion_nat_gradient_random_p_nat={p_nat}.png", dpi=300
        )
        plt.show()

    else:
        raise ValueError("Inapropriate initial state type passed.")

    return


def eq_after_inv(
    width: int,
    p_nat: float,
    ctrl_param: float,
    n_repeats: int = 5,
    resolution: int = 100,
):
    """Creates a plot displaying the presence of the native vegetation
    after introduction of invasive species for the InvasiveVegetation model.

    Args:
        width: The number of cells in one side of the cellular automata grid.
        p_nat: The initial proportion of native species.
        ctrl_param: The control paramater (positive feedback), set for both species.

    Returns:
        Nothing, a graph is saved and shown."""
    density_after = []
    density_after_list = []
    equilibrium_max_cluster = []
    equilibrium_max_cluster_list = []
    equilibrium_cluster_count = []
    equilibrium_cluster_count_list = []
    total_empty = []

    count = 0
    pp_inv = np.linspace(0, 1, resolution)

    for _ in trange(n_repeats):
        vegetation = InvasiveVegetation(
            width, species_prop=(p_nat, 0), control=ctrl_param
        )
        vegetation.run()
        initial_grid = vegetation.grid.copy()
        total_cells = vegetation.area
        total_empty.append(
            (vegetation.area - vegetation.total_alive()) / vegetation.area
        )

        for p_inv in pp_inv:
            # Introduce invasive
            count += 1

            # Introduces invasive species
            vegetation.introduce_invasive(p_inv)
            vegetation.run(iterations=300)

            # Calculates and obtains plotted data
            density_after.append(vegetation.species_alive()[0] / total_cells)
            equilibrium_cluster_count.append(count_clusters(vegetation.grid))
            equilibrium_max_cluster.append(maximum_cluster_size(vegetation.grid))

            # Resets the grid to it's initial state
            vegetation.grid = initial_grid.copy()

        # Appends and resets values to the corresponding lists
        density_after_list.append(density_after)
        density_after = []

        equilibrium_cluster_count_list.append(equilibrium_cluster_count)
        equilibrium_cluster_count = []

        equilibrium_max_cluster_list.append(equilibrium_max_cluster)
        equilibrium_max_cluster = []

    # Initialises the lists as arrays for mean calculations
    density_after_list = np.asarray(density_after_list)
    density_after_avg = density_after_list.mean(axis=0)

    equilibrium_cluster_count_list = np.asarray(equilibrium_cluster_count_list)
    equilibrium_cluster_count_avg = equilibrium_cluster_count_list.mean(axis=0)

    equilibrium_max_cluster_list = np.asarray(equilibrium_max_cluster_list)
    equilibrium_max_cluster_avg = equilibrium_max_cluster_list.mean(axis=0)

    total_empty = np.asarray(total_empty)
    total_empty_avg = total_empty.mean(axis=0)
    p_inv_avg = pp_inv * total_empty_avg

    # Make the plots
    fig, axes = plt.subplots(
        nrows=3, ncols=1, layout="constrained", figsize=(8, 10), sharex=True
    )
    # Plots the equilibrium density after introduction of invasive species
    axes[0].plot(p_inv_avg, density_after_avg)
    axes[0].set_ylabel("Equilibrium Density")
    axes[0].set_ylim(0, max(density_after_avg) + 0.01)

    # Plots the cluster count after introduction of invasive species
    axes[1].plot(p_inv_avg, equilibrium_cluster_count_avg)
    axes[1].set_ylabel("Cluster Count")
    axes[1].set_ylim(0, None)

    # Plots the Giant component after introduction of invasive species
    axes[2].plot(p_inv_avg, equilibrium_max_cluster_avg)
    axes[2].set_ylabel("Giant Component")
    axes[2].set_yscale("log")

    fig.supxlabel("Initial Proportion of Introduced Invasive Species")
    return fig
