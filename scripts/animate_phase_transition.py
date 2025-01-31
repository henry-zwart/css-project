from collections.abc import Sequence

import matplotlib.animation as animation
import numpy as np

from css_project.model import VegetationModel
from css_project.vegetation import Vegetation
from css_project.visualisation import plot_grid


def animate_phase_transition(
    model: VegetationModel, control_values: Sequence[float], fps: int = 60
):
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


def main():
    WIDTH = 128
    control_values = np.linspace(1, 16, 1000)
    control_values = np.concatenate((control_values[::-1], control_values))

    model = Vegetation(WIDTH, alive_prop=0.1, control=16)
    model.run(1000)

    ani = animate_phase_transition(model, control_values, fps=200)
    ani.save("phase_transition_2.mp4")
    #
    # plt.show()


if __name__ == "__main__":
    main()
