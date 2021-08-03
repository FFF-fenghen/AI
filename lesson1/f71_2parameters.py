import f70dataset
import numpy as np
import f70_plot_utils as plt

m = 100
xs, ys = f70dataset.get_beans(m)
plt.show_scatter(xs, ys)

alpha = 0.01
w1 = 0.1
w2 = 0.1
b = 0.1

x1 = xs[:, 0]
x2 = xs[:, 1]


def forward_propagation(x1, x2):
    z = w1 * x1 + w2 * x2 + b
    a = 1 / (1 + np.exp(-z))
    return a


for _ in range(5000):
    for i in range(m):
        x = xs[i]
        y = ys[i]
        x1 = x[0]
        x2 = x[1]
        a = forward_propagation(x1, x2)

        e = (y - a) ** 2
        deda = -2 * (y - a)
        dadz = a * (1 - a)
        dzdw1 = x1
        dzdw2 = x2
        dzdb = 1

        w1 = w1 - alpha * deda * dadz * dzdw1
        w2 = w2 - alpha * deda * dadz * dzdw2
        b = b - alpha * deda * dadz * dzdb

plt.show_scatter_surface(xs, ys, forward_propagation)
