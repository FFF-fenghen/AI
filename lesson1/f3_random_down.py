import f0_dataset
from matplotlib import pyplot as plt
import numpy

test_number = 100
xs, ys = f0_dataset.get_beans(test_number)
plt.xlabel('volume')
plt.ylabel('toxicity')
w = 0.1
# y_pre = w * xs
# plt.scatter(xs, ys)
# plt.plot(xs, y_pre)
# plt.show()

# 随机梯度下降方式
# alpha = 0.05
# for m in range(5):
#     for i in range(test_number):
#         x = xs[i]
#         y = ys[i]
#         k = 2 * x * x * w + (-2 * x * y)
#         w = w - alpha * k
#         plt.clf()
#         y_pre = w * xs
#         plt.xlim(0, 1)
#         plt.ylim(0, 1.3)
#         plt.scatter(xs, ys)
#         plt.plot(xs, y_pre)
#         plt.pause(0.0001)


# 固定步长的方法
num = 0
alpha = 0.05
step = 0.01

for i in range(test_number*2): # 一百份梯度的距离还不一定足够，如果跨度很大的话
    k = 2 * numpy.sum(xs ** 2) * w - 2 * numpy.sum(ys * xs)
    k = k / test_number  # 这一步求的是斜率
    if k > 0:
        w = w - step
    else:
        w = w + step
    plt.clf()
    y_pre = w * xs
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.4)
    plt.scatter(xs, ys)
    plt.plot(xs, y_pre)
    plt.pause(0.001)

# 批量梯度下降
num = 0
alpha = 0.05
step = 0.01
for i in range(test_number):
    k = 2 * numpy.sum(xs ** 2) * w - 2 * numpy.sum(ys * xs)
    k = k / test_number  # 这一步求的是斜率
    w = w - alpha * k
    plt.clf()
    y_pre = w * xs
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.4)
    plt.scatter(xs, ys)
    plt.plot(xs, y_pre)
    plt.pause(0.001)