import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from css_project.complexity import (
    count_clusters,
    fluctuation_cluster_size,
    maximum_cluster_size,
)
from css_project.vegetation import InvasiveVegetation
from css_project.visualisation import animate_ca, plot_grid


def count_states(alive_nat, alive_inv, vegetation, total_cells):
    alive_n, alive_i = vegetation.species_alive()
    alive_nat.append(alive_n / total_cells)
    alive_inv.append(alive_i / total_cells)

    return alive_nat, alive_inv


def plot_proportion(alive_nat, alive_inv):
    # Plot ratio of dead, native, and invasive cells
    iterations = list(range(len(alive_nat)))

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, alive_nat, linestyle="-", label="Native Species")
    plt.plot(iterations, alive_inv, linestyle="-", label="Invasive Species")
    plt.title("Proportion of Native and Invasive species vs Iterations")
    plt.xlabel("Time Step")
    plt.ylabel("Proportion Cells")
    plt.savefig("results/proportion_nat_inv.png", dpi=300)


def run_animation(vegetation, timespan):
    ani = animate_ca(vegetation, timespan)
    ani.save("vegetation.gif")


def run_model(
    width,
    p_nat=0.25,
    p_inv=0.25,
    timespan=40,
    initial_state="random",
    t_eq=20,
    inv_type="random",
):
    vegetation = InvasiveVegetation(width, species_prop=(p_nat, 0.0))
    fig, ax = plot_grid(vegetation)

    fig.savefig("results/veg_grid.png", dpi=300)

    if initial_state == "equilibrium":
        t = 0
        total_cells = vegetation.width * vegetation.width

        alive_nat = []
        alive_inv = []
        alive_nat, alive_inv = count_states(
            alive_nat, alive_inv, vegetation, total_cells
        )

        while t < t_eq:
            vegetation.update()
            alive_nat, alive_inv = count_states(
                alive_nat, alive_inv, vegetation, total_cells
            )
            t += 1

        fig, ax = plot_grid(vegetation)
        fig.savefig("results/veg_grid_before_inv.png", dpi=300)

        # Introduce invasive species
        vegetation.introduce_invasive(p_inv, inv_type)

        # Copy grid to use in animation after introduction of invasive sp
        initial_grid = vegetation.grid.copy()

        t = 0
        while t < timespan:
            vegetation.update()
            alive_nat, alive_inv = count_states(
                alive_nat, alive_inv, vegetation, total_cells
            )
            t += 1

        # plot
        plot_proportion(alive_nat, alive_inv)

        # Reset grid to initial state
        vegetation.grid = initial_grid.copy()

        # Make animation of grid
        run_animation(vegetation, timespan)

    return


def run_model_multiple(
    width,
    p_nat,
    pp_inv,
    timespan=40,
    initial_state="random",
    t_eq=30,
    inv_type="random",
):
    if initial_state == "equilibrium":
        vegetation = InvasiveVegetation(width, species_prop=(p_nat, 0.0))

        # vegetation.initial_grid(type=initial_state)
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
        plt.savefig("results/proportion_nat_gradient_eq.png", dpi=300)
        plt.show()

    elif initial_state == "random":
        vegetation = InvasiveVegetation(width, species_prop=(0.25, 0.25))
        # vegetation.initial_grid(type=initial_state)
        t = 0
        total_cells = vegetation.width * vegetation.width

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
        plt.savefig("results/proportion_nat_gradient_random.png", dpi=300)
        plt.show()

    else:
        raise ValueError("Inapropriate initial state type passed.")

    return


def run_new_model(width, p_nat, p_inv):
    vegetation = InvasiveVegetation(width, species_prop=(p_nat, 0))
    vegetation.run()

    vegetation.introduce_invasive(p_inv)
    ani = animate_ca(vegetation, 200, fps=25)
    ani.save("vegetation_new.gif")


def eq_after_inv_cluster_plots(
    pp_inv,
    density_after,
    equilibrium_cluster_count,
    equilibrium_max_cluster,
    equilibrium_fluctuation,
):
    fig, axes = plt.subplots(
        nrows=4, ncols=1, layout="constrained", figsize=(8, 10), sharex=True
    )
    # Plots the equilibrium density after introduction of invasive species
    axes[0].plot(pp_inv, density_after)
    axes[0].set_ylabel("Equilibrium Density")
    axes[0].set_ylim(0, max(density_after) + 0.01)

    # Plots the cluster count after introduction of invasive species
    axes[1].plot(pp_inv, equilibrium_cluster_count)
    axes[1].vlines(pp_inv, ymin=0, ymax=equilibrium_cluster_count)
    axes[1].set_ylabel("Cluster Count")
    axes[1].set_ylim(0, None)

    # Plots the Giant component after introduction of invasive species
    axes[2].plot(pp_inv, equilibrium_max_cluster)
    axes[2].vlines(pp_inv, ymin=0, ymax=equilibrium_max_cluster)
    axes[2].set_ylabel("Giant Component")
    axes[2].set_yscale("log")

    # Plots the fluctuation of the cluster sizes after introduction
    # of invasive species
    axes[3].plot(pp_inv, equilibrium_fluctuation)
    axes[3].vlines(pp_inv, ymin=0, ymax=equilibrium_fluctuation)
    axes[3].set_ylabel("Cluster size fluctuation")
    axes[3].set_yscale("log")

    fig.supxlabel("Introduced Proportion ofInvasive Species on Dead Cells ")

    plt.show()

    return


def eq_after_inv(width, p_nat):
    density_after = []
    density_after_list = []
    equilibrium_max_cluster = []
    equilibrium_max_cluster_list = []
    equilibrium_fluctuation = []
    equilibrium_fluctuation_list = []
    equilibrium_cluster_count = []
    equilibrium_cluster_count_list = []

    count = 0
    pp_inv = np.linspace(0, 1, 100)

    for _ in range(0, 5):
        vegetation = InvasiveVegetation(width, species_prop=(p_nat, 0))
        vegetation.run()
        initial_grid = vegetation.grid.copy()
        total_cells = vegetation.area

        for p_inv in pp_inv:
            # Introduce invasive
            count += 1
            print("Count: ", count)

            vegetation.introduce_invasive(p_inv)
            vegetation.run(iterations=500)

            density_after.append(vegetation.species_alive()[0] / total_cells)
            equilibrium_cluster_count.append(count_clusters(vegetation.grid))
            equilibrium_max_cluster.append(maximum_cluster_size(vegetation.grid))
            equilibrium_fluctuation.append(fluctuation_cluster_size(vegetation.grid))

            vegetation.grid = initial_grid.copy()

        density_after_list.append(density_after)
        density_after = []

        equilibrium_cluster_count_list.append(equilibrium_cluster_count)
        equilibrium_cluster_count = []

        equilibrium_max_cluster_list.append(equilibrium_max_cluster)
        equilibrium_max_cluster = []

        equilibrium_fluctuation_list.append(equilibrium_fluctuation)
        equilibrium_fluctuation = []

    density_after_list = np.asarray(density_after_list)
    density_after_avg = density_after_list.mean(axis=0)

    equilibrium_cluster_count_list = np.asarray(equilibrium_cluster_count_list)
    equilibrium_cluster_count_avg = equilibrium_cluster_count_list.mean(axis=0)

    equilibrium_max_cluster_list = np.asarray(equilibrium_max_cluster_list)
    equilibrium_max_cluster_avg = equilibrium_max_cluster_list.mean(axis=0)

    equilibrium_fluctuation_list = np.asarray(equilibrium_fluctuation_list)
    equilibrium_fluctuation_avg = equilibrium_fluctuation_list.mean(axis=0)

    eq_after_inv_cluster_plots(
        pp_inv,
        density_after_avg,
        equilibrium_cluster_count_avg,
        equilibrium_max_cluster_avg,
        equilibrium_fluctuation_avg,
    )


if __name__ == "__main__":
    timespan = 20
    width = 128
    p_nat = 0.25
    p_inv = 0.85

    run_new_model(width, p_nat, p_inv)

    # vegetation = InvasiveVegetation(width, species_prop=(p_nat, 0))
    # vegetation.run(iterations=2)
    # vegetation.introduce_invasive(p_inv)
    # vegetation.run(iterations=1)

    # eq_after_inv(width, p_nat)

    # runs = 5
    # run_new_model(width, species_prop=(0.25, 0.25))

    # pp_inv = np.linspace(0, 1, runs, endpoint=False)
    # run_model_multiple(
    #     width, 0.25, pp_inv, timespan=timespan, initial_state="equilibrium"
    # )
