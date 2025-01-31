import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.colors as colors
import matplotlib.colors as mpc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from css_project.complexity import (
    Compression,
    avg_cluster_size,
    compressed_size,
    count_clusters,
    fluctuation_cluster_size,
    maximum_cluster_size,
)
from css_project.logistic import Logistic
from css_project.vegetation import Vegetation
from css_project.visualisation import QUALITATIVE_COLOURS


def init_and_run_model(
    model_type,
    width: int,
    control: float,
    alive_prop: float,
    n_steps: int,
    alive_prop_idx: int,
    control_idx: int,
):
    model = model_type(width, control=control, alive_prop=alive_prop)
    model.run(n_steps)
    results = (
        model.proportion_alive_list[-1],
        compressed_size(model.grid, Compression.ZLIB),
        count_clusters(model.grid),
        avg_cluster_size(model.grid),
        maximum_cluster_size(model.grid),
        fluctuation_cluster_size(model.grid),
    )
    return alive_prop_idx, control_idx, results


def make_heatmaps(
    model_type,
    width: int,
    initial_densities: list[float] | np.ndarray,
    control_values: list[float] | np.ndarray,
    init_steps: int = 200,
    control_value_name: str = "Control value",
):
    initial_densities = np.asarray(initial_densities)
    control_values = np.asarray(control_values)

    alive_grid = np.empty(
        (len(control_values), len(initial_densities)), dtype=np.float64
    )
    complexity_grid = np.empty_like(alive_grid)
    cluster_count = np.empty_like(alive_grid)
    avg_clusters = np.empty_like(alive_grid)
    max_clusters = np.empty_like(alive_grid)
    fluctuation_clusters = np.empty_like(alive_grid)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                init_and_run_model,
                model_type,
                width,
                c,
                p,
                init_steps,
                p_i,
                c_i,
            )
            for p_i, p in enumerate(initial_densities)
            for c_i, c in enumerate(control_values)
        ]

        for future in tqdm(
            as_completed(futures), total=len(control_values) * len(initial_densities)
        ):
            p_i, c_i, results = future.result()
            (
                prop_alive,
                complexity,
                n_clusters,
                avg_cluster,
                max_cluster,
                fluctuation,
            ) = results
            alive_grid[c_i, p_i] = prop_alive
            complexity_grid[c_i, p_i] = complexity
            cluster_count[c_i, p_i] = n_clusters
            avg_clusters[c_i, p_i] = avg_cluster
            max_clusters[c_i, p_i] = max_cluster
            fluctuation_clusters[c_i, p_i] = fluctuation

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(8, 6),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    data = [
        alive_grid,
        cluster_count,
        max_clusters,
        fluctuation_clusters,
    ]
    log_scale = [False, False, True, True]
    X, Y = np.meshgrid(initial_densities, control_values)
    for d, do_log, ax in zip(data, log_scale, axes.flatten(), strict=True):
        norm = colors.LogNorm() if do_log else None
        hm = ax.pcolormesh(X, Y, d, norm=norm)
        ax.set_xscale("log")
        fig.colorbar(hm, ax=ax)

    axes[0, 0].set_title("Cell density")
    axes[0, 1].set_title("Cluster count")
    axes[1, 0].set_title("Maximum size of a cluster")
    axes[1, 1].set_title("Cluster size fluctuation")

    # Make giant component and fluctuation colours logscale
    fig.supxlabel("Initial population density")
    fig.supylabel(control_value_name)

    # Produce phase plot by identifying qualitative regions of heatmaps
    phases = np.zeros_like(alive_grid, dtype=np.int64)
    zero_clusters = cluster_count == 0
    one_cluster = cluster_count == 1
    isolated_plants = (cluster_count >= 1) & (avg_clusters < 2)
    has_clusters = (cluster_count >= 1) & (avg_clusters >= 2)
    # few_clusters = (cluster_count >= 1) & (cluster_count <= 50)
    # several_clusters = cluster_count > 50
    giant_component_exists = max_clusters > 1e-1
    low_fluctuation = fluctuation_clusters <= 1e-4
    high_fluctuation = fluctuation_clusters > 1e-4
    nontrivial_density = alive_grid >= 0.1

    # No clusters; empty grid
    #   - n_clusters == 0
    phases[zero_clusters] = 0

    # Sparse nonuniformly-distributed clusters
    #   - Small number of clusters
    #   - Low cluster size fluctuation (e.g. < 10^-3)
    phases[isolated_plants & low_fluctuation] = 1
    # phases[few_clusters & low_fluctuation] = 1

    # 'Spots' phase
    #   - Large number of clusters
    #   - Low cluster size fluctuation (e.g. < 10^-3)
    phases[has_clusters & low_fluctuation] = 2
    # phases[several_clusters & low_fluctuation] = 2

    # 'Labyrinthes' phase
    #   - Large number of clusters
    #   - Moderate-to-high cluster size fluctuation (e.g. > 10^-3)
    phases[has_clusters & high_fluctuation] = 3

    # Giant component/percolating cluster
    #   - Large max cluster (e.g. > 0.1)
    phases[giant_component_exists] = 4

    # Dense vegetation
    #   - Single cluster
    #   - Nontrivial density (e.g. > 0.1)
    phases[one_cluster & nontrivial_density] = 5

    phase_fig, phase_ax = plt.subplots(layout="constrained")

    # Create colourmap
    # https://stackoverflow.com/questions/52842553/matplotlib-listedcolormap-not-mapping-colors
    u = np.unique(phases)
    bounds = np.concatenate(
        ([phases.min() - 1], u[:-1] + np.diff(u) / 2.0, [phases.max() + 1])
    )
    norm = colors.BoundaryNorm(bounds, len(bounds) - 1)
    palette = [QUALITATIVE_COLOURS[i] for i in sorted(np.unique(phases))]
    cmap = mpc.ListedColormap(palette)

    hm = phase_ax.pcolormesh(X, Y, phases, cmap=cmap, norm=norm)
    phase_ax.set_xscale("log")

    phase_fig.supxlabel("Initial density")
    phase_fig.supylabel("Control variable")
    phase_fig.suptitle("Phase plot")

    phase_labels = {
        0: "No clusters (empty grid)",
        1: "Sparse clusters",
        2: "Spots",
        3: "Labyrinths",
        4: "Giant component",
        5: "Dense vegetation",
    }

    observed_states = sorted(np.unique(phases))
    legend_patches = [
        mpatches.Patch(color=QUALITATIVE_COLOURS[k], label=phase_labels[k])
        for k in observed_states
    ]
    phase_ax.legend(handles=legend_patches, loc="upper right", title="Phases")

    return fig, phase_fig


def main(width: int, resolution: int):
    # Prepare states
    positives = np.linspace(start=1, stop=20, num=resolution)
    initial_probs = np.logspace(start=-6, stop=0, num=resolution)

    fig, phase = make_heatmaps(
        Vegetation,
        width,
        initial_probs,
        positives,
        init_steps=500,
        control_value_name="Positive feedback weight",
    )
    fig.savefig("results/heatmaps_all_native_vegetation.png", dpi=500)
    phase.savefig("results/phaseplot_native_vegetation.png", dpi=500)

    nutrient_level = np.linspace(start=0.001, stop=70, num=resolution)

    fig, phase = make_heatmaps(
        Logistic,
        width,
        initial_probs,
        nutrient_level,
        init_steps=1000,
        control_value_name="Nutrient availability",
    )

    fig.savefig("results/heatmaps_all_native_logistic.png", dpi=500)
    phase.savefig("results/phaseplot_native_logistic.png", dpi=500)


if __name__ == "__main__":
    QUICK_WIDTH = 128
    QUICK_RESOLUTION = 15
    FULL_WIDTH = 256
    FULL_RESOLUTION = 80

    parser = argparse.ArgumentParser(
        prog="Heatmaps", description="Generate heatmaps and phase diagrams"
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        main(width=QUICK_WIDTH, resolution=QUICK_RESOLUTION)
    else:
        main(width=FULL_WIDTH, resolution=FULL_RESOLUTION)
