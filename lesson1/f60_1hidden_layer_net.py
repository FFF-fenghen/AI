import numpy as np
import matplotlib.pyplot as plt
import f50_dataset


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_propagation(x):
    z1_1 = w11_1 * x + b1_1
    a1_1 = sigmoid(z1_1)
    z2_1 = w21_2 * x + b2_1
    a2_1 = sigmoid(-z2_1)
    z1_2 = w11_1 * a1_1 + w21_2 * a2_1 + b2_1
    a1_2 = sigmoid(z1_2)
    return z1_1, z2_1, a1_1, a2_1, z1_2, a1_2


m = 100
xs, ys = f50_dataset.get_beans(m)
plt.title('hill figure')
plt.xlabel('volume')
plt.ylabel('toxicity')
plt.scatter(xs, ys)


alpha = 0.03
w11_1 = np.random.rand()
w11_2 = np.random.rand()
w12_1 = np.random.rand()
w21_2 = np.random.rand()
b1_1 = np.random.rand()
b2_1 = np.random.rand()
b1_2 = np.random.rand()

z1_1, z2_1, a1_1, a2_1, z1_2, a1_2 = forward_propagation(xs)
plt.plot(xs, a1_2)
plt.show()
for n in range(5000):
    for i in range(m):
        x = xs[i]
        y = ys[i]
        z1_1, z2_1, a1_1, a2_1, z1_2, a1_2 = forward_propagation(x)
        e = (y - a1_2)**2
        deda1_2 = -2 * (y - a1_2)
        da1_2dz1_2 = a1_2 *(1-a1_2)

        dz1_2dw11_2 = a1_1
        dz1_2dw21_2 = a2_1
        dz1_2db1_2 = 1

        # 求导第一层
        dz1_2da1_1 = w11_2
        dz1_2da2_1 = w21_2
        deda1_1 = deda1_2 * da1_2dz1_2 * dz1_2da1_1
        deda2_1 = deda1_2 * da1_2dz1_2 * dz1_2da2_1

        da1_1dz1_1 = a1_1*(1-a1_1)
        da2_1dz2_1 = a2_1*(1-a2_1)

        dz1_1db1_1 = 1
        dz1_1dw11_1 = x
        dz2_1dw12_1 = x
        dz1_1db1_1 = 1

        # 调整w11_2, w21_2, b2_1 的值
        dedw11_2 = deda1_2 * da1_2dz1_2 *dz1_2dw11_2
        w11_2 = w11_2 - alpha * dedw11_2
        dedw21_2 = deda1_2 * da1_2dz1_2 * dz1_2dw21_2
        w21_2 = w21_2 - alpha * dedw21_2
        dedb1_2 = deda1_2 * da1_2dz1_2 * dz1_2db1_2
        b2_1 = b2_1 - alpha * dedb1_2

        # 调整参数
        dedb1_1 = deda1_1 * da1_1dz1_1 * dz1_1db1_1
        b1_1 = b1_1 - alpha *dedb1_1

        dedw11_1 = deda1_1 * da1_1dz1_1 * dz1_1dw11_1
        w11_1 = w11_1 - alpha * dedw11_1

        dedw12_1 = deda2_1 * da2_1dz2_1 * dz2_1dw12_1
        w12_1 = w12_1 - alpha * dedw12_1

        dedb2_1 = deda2_1 * da2_1dz2_1 * dz1_1db1_1
        b2_1 = b2_1 - alpha * dedb2_1

    if n % 100 == 0:
        plt.clf()
        plt.scatter(xs,ys)
        z1_1, z2_1, a1_1, a2_1, z1_2, a1_2 = forward_propagation(xs)
        plt.plot(xs, a1_2)
        plt.pause(0.001)
print('w11_1,w11_2, w12_1, w21_2, b1_1, b2_1, b1_2', w11_1,w11_2, w12_1, w21_2, b1_1, b2_1, b1_2)

# what id 






