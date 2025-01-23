import pytest

from css_project.fine_grained import FineGrained


@pytest.mark.parametrize(
    "nutrient_level,nutrient_consume_rate",
    [
        (None, None),
        (0.0, 0.0),
        (-0.0, 0.0),
        (0.5, 0.5),
        (1.0, None),
    ],
)
def test_valid_nutrient_level(nutrient_level, nutrient_consume_rate):
    kwargs = {}
    if nutrient_level is not None:
        kwargs["nutrient_level"] = nutrient_level
    if nutrient_consume_rate is not None:
        kwargs["nutrient_consume_rate"] = nutrient_consume_rate

    FineGrained(**kwargs)


@pytest.mark.parametrize("nutrient_level", [-1.0, -0.001])
def test_invalid_nutrient_level(nutrient_level):
    with pytest.raises(ValueError):
        FineGrained(nutrient_level=nutrient_level)


@pytest.mark.parametrize("nutrient_consume_rate", [0.0, -0.0, 0.5, 1.0])
def test_valid_nutrient_consumption(nutrient_consume_rate):
    _ = FineGrained(nutrient_consume_rate=nutrient_consume_rate)


@pytest.mark.parametrize(
    "nutrient_level,nutrient_consume_rate",
    [
        (None, -1.0),
        (None, -0.001),
        (0.5, 0.6),
    ],
)
def test_invalid_nutrient_consumption(nutrient_level, nutrient_consume_rate):
    kwargs = {}
    if nutrient_level is not None:
        kwargs["nutrient_level"] = nutrient_level
    if nutrient_consume_rate is not None:
        kwargs["nutrient_consume_rate"] = nutrient_consume_rate

    with pytest.raises(ValueError):
        FineGrained(**kwargs)
