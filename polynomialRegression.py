"""
finding the line parameterized by some set of slope and y intercept parameters to produce one up which is the
prediction for the unlabelled test inputs
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
np.random.seed(0)

points = 500
X = np.linspace(-3, 3, points)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, points)

# plotting random generated points
plt.scatter(X, y)

model = Sequential()
model.add(Dense(50, activation='sigmoid', input_dim=1))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1))
adam = Adam(lr=0.1)
model.compile(loss='mse', optimizer=adam)
model.fit(X, y, epochs=50)
predictions = model.predict(X)
plt.scatter(X,y)
plt.plot(X, predictions, 'ro')
plt.show()
