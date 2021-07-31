import f40_dataset
from matplotlib import pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D

test_number = 100
xs, ys = f40_dataset.get_beans(test_number)

# 绘制散点图
w = 0.1
b = 0.1
alpha = 0.01
y_pre = w * xs + b
plt.title('beans')
plt.xlabel('volume')
plt.ylabel('toxicity')
plt.xlim(0, 1.1)
plt.ylim(0, 1.4)
plt.scatter(xs, ys)
plt.plot(xs, y_pre)
plt.show()

# 随机梯度下降
for n in range(500):
    for i in range(test_number):
        x = xs[i]
        y = ys[i]
        dw = 2 * x ** 2 * w + 2 * x * b - 2 * x * y
        db = 2 * b + 2 * x * w - 2 * y
        w = w - alpha * dw
        b = b - alpha * db
    y_pre = w * xs + b
    plt.clf()
    plt.scatter(xs, ys)
    plt.xlim(0, 1)
    plt.ylim(0, 1.2)
    plt.plot(xs, y_pre)
    plt.pause(0.001)

