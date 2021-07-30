import f0_dataset
from matplotlib import pyplot

test_numer = 100
xs, ys = f0_dataset.get_beans(test_numer)
print(xs)
print(ys)

# 配置图像
pyplot.title('Size_Toxicity', fontsize=12)
pyplot.xlabel('Bean size')
pyplot.ylabel('Toxicity')
pyplot.scatter(xs, ys)


# Rossonblatt sensor
alpha = 0.05
w = 0.5
for m in range(test_numer):  # 增加学习次数，提高拟合度
    for i in range(test_numer):
        x = xs[i]
        y = ys[i]
        y_pre = w * x
        e = y - y_pre
        w = w + alpha * w * e
       # sleep()

# 绘制一元函数图像
y_pre = w*xs
pyplot.plot(xs, y_pre)
pyplot.show()

