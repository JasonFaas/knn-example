import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the nba player of week dataset
data = pd.read_csv('NBA_player_of_the_week.csv')

def set_height_inches(row):
    height_ = row['Height']
    if "-" in height_:
        split = height_.strip().split("-")
        return float(split[0]) * 12 + float(split[1])
    else:
        split = height_.strip().split("cm")
        return float(split[0]) / 2.54

def set_weight_lbs(row):
    weight_ = row['Weight']
    kg = "kg"
    if kg in weight_:
        return float(weight_.split(kg)[0]) * 2.2
    else:
        return float(weight_)


data['height_inches'] = data.apply(set_height_inches, axis=1)
data['weight_lbs'] = data.apply(set_weight_lbs, axis=1)

data_x = data['height_inches']
data_y = data['weight_lbs']

x_train, x_test, y_train, y_test = train_test_split(data_x.values, data_y.values, train_size=0.9, random_state=33)
# print(type(x_train))
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test.values)
x_train, x_test, y_train, y_test = x_train.reshape(-1, 1), x_test.reshape(-1, 1), y_train.reshape(-1, 1), y_test.reshape(-1, 1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)
print("Diabetes_y_pred")
print(y_pred)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(x_train, y_train,  color='green', alpha=0.5)
plt.scatter(x_test, y_test,  color='red', alpha=0.5)
plt.plot(x_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()