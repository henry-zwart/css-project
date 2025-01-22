from css_project.simple_ca import GameOfLife


def test_create_game_of_life():
    model = GameOfLife(width=64)
    assert model is not None


def test_game_of_life_properties():
    model = GameOfLife(width=64)
    assert model.width == 64, "unexpected width"
    assert model.grid.sum() == 0, "expected empty grid"
