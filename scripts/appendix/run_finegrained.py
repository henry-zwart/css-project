import argparse

from css_project.fine_grained import FineGrained

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FineGrained", description="Run finegrained model"
    )
    parser.add_argument("--width", type=int)
    args = parser.parse_args()

    model = FineGrained(
        args.width,
        nutrient_level=1.0,
        nutrient_diffusion_rate=0.21,  # 0.225
        nutrient_consume_rate=0.1,
        nutrient_regenerate_rate=0.8,
    )
    model.initial_grid(p=0.1)

    for _ in range(100):
        model.update()
