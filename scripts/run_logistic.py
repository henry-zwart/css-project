import argparse

from css_project.logistic import Logistic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Logistic", description="Run logistic model")
    parser.add_argument("--width", type=int)
    args = parser.parse_args()

    model = Logistic(
        args.width,
        consume_rate=63.7,
        control=35,
        alive_prop=0.001,
        random_seed=42,
    )

    model.run(1000)
