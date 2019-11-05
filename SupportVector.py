import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
inputScalar = StandardScaler()
X = inputScalar.fit_transform(X)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

prediction = regressor.predict(inputScalar.transform(np.array(X)))

plt.scatter(X, y, color='red')
plt.plot(X, y, color='green')
plt.plot(X, prediction, color='blue')
plt.show()