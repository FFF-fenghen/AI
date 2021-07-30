import f0_dataset
from matplotlib import pyplot
import numpy

test_numer = 100
xs, ys = f0_dataset.get_beans(test_numer)
print(xs)
print(ys)

# 配置图像
pyplot.title('Size_Toxicity', fontsize=12)
pyplot.xlabel('Bean size')
pyplot.ylabel('Toxicity')
pyplot.scatter(xs, ys)

# 绘制函数
ws = numpy.arange(0, 3, 0.1)
es = []
for w in ws:
    y_pre = w*xs
    e = (ys - y_pre)**2
    sum_e = numpy.sum(e) / test_numer
    es.append(sum_e)

w_min = numpy.sum(xs*ys) / numpy.sum(xs**2)
print('min:' + str(w_min))

# W 取值和 误差e之间的关系图像
# pyplot.title('cost_function')
# pyplot.xlabel('w')
# pyplot.ylabel('e')
# pyplot.plot(ws, es)
# # pyplot.plot(xs, y_pre)
# pyplot.show()


# 使用最低点误差作为参数w进行预测
y_pre = w_min * xs
pyplot.title('lowest point to predict')
pyplot.scatter(xs, ys)
pyplot.xlabel('xs')
pyplot.ylabel('ys')
pyplot.plot(xs, y_pre)
pyplot.show()


