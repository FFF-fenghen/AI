import matplotlib.pyplot as plt
import f50_dataset
import numpy as np

num = 100
xs, ys = f50_dataset.get_beans(num)
plt.table('Size-toxicity function', fontsize=12)
plt.xlabel('volume')
plt.ylabel('toxicity')
plt.scatter(xs, ys)

w = 0.1
b = 0.1
z = w * xs + b
a = 1 / (1 + np.exp(-z))
alpha = 0.01
for n in range(10000):
    for i in range(num):
        x = xs[i]
        y = ys[i]

        z = w * x + b
        a = 1 / (1 + np.exp(-z))
        e = (y - a) ** 2

        de_da = -2 * (y - a)
        da_dz = a * (1 - a)
        dz_dw = x

        de_dw = de_da * da_dz * dz_dw
        de_db = de_da * da_dz

        w = w - alpha * de_dw
        b = b - alpha * de_db
    if n % 100 == 0:
        plt.clf()
        plt.scatter(xs, ys)
        z = w * xs + b
        a = 1 / (1 + np.exp(-z))
        plt.xlim(0, 1)
        plt.ylim(0, 1.2)
        plt.plot(xs, a)
        plt.pause(0.01)
