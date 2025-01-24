import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from css_project.vegetation import InvasiveVegetation
from css_project.visualisation import animate_ca, plot_grid


def count_states(alive_nat, alive_inv, vegetation, total_cells):
    alive_n, alive_i = vegetation.total_alive()
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
    plt.savefig("proportion_nat_inv.png", dpi=300)


def run_animation(vegetation, timespan):
    ani = animate_ca(vegetation, timespan)
    ani.save("vegetation.gif")


def run_model(
    vegetation,
    p_nat=0.25,
    p_inv=0.25,
    timespan=40,
    initial_state="random",
    t_eq=20,
    inv_type="random",
):
    vegetation.initial_grid(type=initial_state)
    fig, ax = plot_grid(vegetation)

    fig.savefig("veg_grid.png", dpi=300)

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
        fig.savefig("veg_grid_before_inv.png", dpi=300)

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
    vegetation,
    p_nat,
    pp_inv,
    timespan=40,
    initial_state="random",
    t_eq=30,
    inv_type="random",
):
    if initial_state == "equilibrium":
        vegetation.initial_grid(type=initial_state)
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
        plt.savefig("proportion_nat_gradient_eq.png", dpi=300)
        plt.show()

    elif initial_state == "random":
        vegetation.initial_grid(type=initial_state)
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
        plt.savefig("proportion_nat_gradient_random.png", dpi=300)
        plt.show()

    else:
        raise ValueError("Inapropriate initial state type passed.")

    return


def heatmap_final_proportions(runs, width):
    # invase vs native on x and y
    # final proportion of native (normalized), 0 is green
    # green what it was at equilibrium

    # divide values by initial proportion

    t_eq = 20  # Change with steady state implementation
    p_values = np.linspace(0, 1, runs)

    for n_value in p_values:
        vegetation = InvasiveVegetation(width)

        vegetation.initial_grid(p_nat=n_value, type="equilibrium")

        total_cells = vegetation.width * vegetation.width
        alive_nat = []
        alive_inv = []
        alive_nat, alive_inv = count_states(
            alive_nat, alive_inv, vegetation, total_cells
        )

        t = 0
        while t < t_eq:
            vegetation.update()
            t += 1
        alive_nat, alive_inv = count_states(
            alive_nat, alive_inv, vegetation, total_cells
        )

        initial_grid = vegetation.grid.copy()

        for i_value in p_values:
            # p_nat = n_value
            # p_inv = i_value

            # Copy grid to use for different invasive concentrations

            # Introduce invasive species
            vegetation.introduce_invasive(i_value, inv_type="random")

            t = 0
            while t < timespan:
                vegetation.update()
                t += 1
            alive_nat, alive_inv = count_states(
                alive_nat, alive_inv, vegetation, total_cells
            )

            # Reset grid to initial state
            vegetation.grid = initial_grid.copy()


if __name__ == "__main__":
    timespan = 10
    width = 64
    runs = 5

    vegetation = InvasiveVegetation(width)
    # run_model(vegetation, initial_state='equilibrium')

    pp_inv = np.linspace(0, 1, runs, endpoint=False)
    run_model_multiple(
        vegetation, 0.25, pp_inv, timespan=timespan, initial_state="equilibrium"
    )
