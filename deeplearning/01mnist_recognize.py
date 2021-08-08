from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train" + str(x_train.shape))
print("y_train" + str(y_train.shape))
plt.imshow(x_train[0], cmap='gray')
plt.show()

x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0

y_train = to_categorical(y_train, 10) # 10 表示又是个数输出结果
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=784))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=10, activation='softmax')) # softmax函数可以让10个数据的概率综合为1

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500, batch_size=128)
