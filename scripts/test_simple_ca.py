from css_project.simple_ca import GameOfLife

a = GameOfLife(3)
a.initial_grid(0.5)
print(a.grid)
a.update()
print(a.grid)
