import f40_dataset
from matplotlib import pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D

test_number = 100
xs, ys = f40_dataset.get_beans(test_number)


w = 0.1
b = 0.1
y_pre = w * xs + b
plt.title('beans')
plt.xlabel('volume')
plt.ylabel('toxicity')
plt.xlim(0, 1.1)
plt.ylim(0, 1.4)
plt.scatter(xs, ys)
plt.plot(xs, y_pre)
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.set_zlim(0, 2)
ax.set_xlabel('ws')
ax.set_ylabel('es')
ax.set_zlabel('b')
w = 0.1
bs = numpy.arange(-2, 2, 0.01)
ws = numpy.arange(-1, 2, 0.1)

for b in bs:
    es = []
    for w in ws:
        y_pre = w * xs + b
        e = numpy.sum((ys - y_pre)**2) / test_number
        es.append(e)
    # plt.plot(ws, es)
    ax.plot(ws, es, b, zdir='y')
plt.show()
