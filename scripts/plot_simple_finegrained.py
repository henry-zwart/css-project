from css_project.fine_grained import FineGrained
from css_project.visualisation import animate_ca, animate_nutrients

if __name__ == "__main__":
    width = 128
    model = FineGrained(
        width,
        nutrient_level=1.0,
        nutrient_diffusion_rate=0.15,
        nutrient_consume_rate=0.025,
    )
    model.initial_grid(p=0.1)
    initial_state = model.grid.copy()

    ani = animate_nutrients(model, 30, fps=2)
    ani.save("nutrient_diffusion.mp4")

    model.grid = initial_state
    ani = animate_ca(model, 30, fps=2)
    ani.save("fine_grained_vegetation_spread.mp4")
