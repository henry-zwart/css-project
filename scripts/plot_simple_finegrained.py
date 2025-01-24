from css_project.fine_grained import FineGrained
from css_project.visualisation import animate_ca, animate_nutrients

if __name__ == "__main__":
    width = 256

    model = FineGrained(
        width,
        nutrient_level=1.0,
        nutrient_diffusion_rate=0.21,  # 0.225
        nutrient_consume_rate=0.1,
        nutrient_regenerate_rate=0.8,
        random_seed=None,
    )
    model.initial_grid(p=0.1)

    ani = animate_nutrients(model, 600, fps=20)
    ani.save("nutrient_diffusion.mp4")

    model.reset()
    ani = animate_ca(model, 600, fps=20)
    ani.save("fine_grained_vegetation_spread.mp4")
