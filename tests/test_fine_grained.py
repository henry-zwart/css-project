import pytest

from css_project.fine_grained import FineGrained


def test_valid_nutrient_level():
    _ = FineGrained()
    _ = FineGrained(nutrient_level=0.0, nutrient_consume_rate=0.0)
    _ = FineGrained(nutrient_level=-0.0, nutrient_consume_rate=0.0)
    _ = FineGrained(nutrient_level=0.5, nutrient_consume_rate=0.5)
    _ = FineGrained(nutrient_level=1.0)


def test_invalid_nutrient_level():
    with pytest.raises(ValueError):
        _ = FineGrained(nutrient_level=-1.0)
    with pytest.raises(ValueError):
        _ = FineGrained(nutrient_level=-0.001)
    with pytest.raises(ValueError):
        _ = FineGrained(nutrient_level=1.001)
    with pytest.raises(ValueError):
        _ = FineGrained(nutrient_level=2.0)


def test_valid_nutrient_consumption():
    _ = FineGrained()
    _ = FineGrained(nutrient_consume_rate=0.0)
    _ = FineGrained(nutrient_consume_rate=-0.0)
    _ = FineGrained(nutrient_consume_rate=0.5)
    _ = FineGrained(nutrient_consume_rate=1.0)


def test_invalid_nutrient_consumption():
    with pytest.raises(ValueError):
        _ = FineGrained(nutrient_consume_rate=-1.0)
    with pytest.raises(ValueError):
        _ = FineGrained(nutrient_consume_rate=-0.001)
    with pytest.raises(ValueError):
        _ = FineGrained(nutrient_consume_rate=1.001)
    with pytest.raises(ValueError):
        _ = FineGrained(nutrient_consume_rate=2.0)
    with pytest.raises(ValueError):
        _ = FineGrained(nutrient_consume_rate=0.6, nutrient_level=0.5)
