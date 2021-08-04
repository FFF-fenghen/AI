import f80dataset
import f80plot_utils as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

m=100
X, Y = f80dataset.get_beans(m)
plt.show_scatter(X, Y)

model = Sequential()
model.add(Dense(units=8, activation='relu', input_dim=2))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
# model.add(Dense(units=8, activation='relu'))
# model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.03), metrics=['accuracy'])
model.fit(X, Y, epochs=9000, batch_size=10)
plt.show_scatter_surface(X, Y, model)