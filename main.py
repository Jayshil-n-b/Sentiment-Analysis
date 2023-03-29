from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import math
import pickle

raw_data = pd.read_csv('./data.csv')

x = raw_data[['bedrooms', 'bathrooms', 'area']]
y = raw_data[['price']]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=40)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("MAE", metrics.mean_absolute_error(y_test, predictions))
print("MSE", math.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Score", model.score(x_test, y_test))

pickle.dump(model, open('model.pkl', 'wb'))
