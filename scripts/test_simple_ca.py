from css_project.simple_ca import GameOfLife

a = GameOfLife(10)
a.initial_grid(0.1)
print(a.grid)

a.update()
