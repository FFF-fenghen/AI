import f72dataset
import numpy as np
import f72_plot_utils as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

m = 100

#简单的sigmoid函数图像
# X, Y = f72dataset.get_beans1(m)
# plt.show_scatter(X, Y)

# model = Sequential()
# model.add(Dense(units=1, activation='sigmoid', input_dim=1))
# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# model.fit(X, Y, epochs=5000, batch_size=1)
# pres = model.predict(X)
# plt.show_scatter_curve(X, Y, pres)

# 山丘 函数图像
# X, Y = f72dataset.get_beans2(m)
# plt.show_scatter(X, Y)
# model = Sequential()
# model.add(Dense(units=2, activation='sigmoid', input_dim=1))
# model.add(Dense(units=1, activation='sigmoid'))
#
# model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.05), metrics=['accuracy'])
# model.fit(X, Y, epochs=5000, batch_size=10)
# pres = model.predict(X)
# plt.show_scatter_curve(X, Y, pres)

#
# # 三维曲面图像
# X, Y = f72dataset.get_beans(m)
# plt.show_scatter(X, Y)
# model = Sequential()
# model.add(Dense(units=1, activation='sigmoid', input_dim=2))
#
# model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.05), metrics=['accuracy'])
# model.fit(X, Y, epochs=5000, batch_size=10)
# pres = model.predict(X)
#
# plt.show_scatter_surface(X, Y, model)



# 三维扭曲曲面图像
X, Y = f72dataset.get_beans4(m)
plt.show_scatter(X, Y)
model = Sequential()
model.add(Dense(units=2, activation='sigmoid', input_dim=2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.05), metrics=['accuracy'])
model.fit(X, Y, epochs=5000, batch_size=1)
pres = model.predict(X)

plt.show_scatter_surface(X, Y, model)