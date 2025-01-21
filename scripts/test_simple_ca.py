from css_project.simple_ca import GameOfLife

a = GameOfLife(10)
a.initial_grid(0.5)
print(a.grid)
print(a.total_alive())

t = 0
while t < 100:
    a.update()
    t += 1

print(a.total_alive())
print(a.grid)
