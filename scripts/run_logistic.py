import argparse

from css_project.logistic import LogisticTwoNative

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Logistic", description="Run logistic model")
    parser.add_argument("--width", type=int)
    args = parser.parse_args()

    model = LogisticTwoNative(
        args.width,
        consume_rate_1=63.7,
        consume_rate_2=63.7,
        control=35,
        species_prop=[0.0005, 0.0005],
        random_seed=42,
    )

    for _ in range(1000):
        model.update()
