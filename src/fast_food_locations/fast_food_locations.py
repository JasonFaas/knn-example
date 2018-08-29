import pandas as pd
import operator
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, accuracy_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#import data and basic filtering
ffl_data = pd.read_csv('FastFoodRestaurants.csv')
main_columns = ['latitude', 'longitude', 'name']
ffl_data = ffl_data[main_columns]
ffl_data = ffl_data.dropna()

#Remove regions outside contenintal US, including bad data
ffl_data = ffl_data.drop(ffl_data[ffl_data['longitude'] > -60].index)
ffl_data = ffl_data.drop(ffl_data[ffl_data['longitude'] < -130].index)
ffl_data = ffl_data.drop(ffl_data[ffl_data['latitude'] < 22].index)


#utilize only top 2 restaurants
name_dict = {}
for ffl_lab, ffl_row in ffl_data.iterrows():
    name_dict[ffl_row['name']] = name_dict.get(ffl_row['name'], 0) + 1
name_sorted_desc = sorted(name_dict.items(), key=operator.itemgetter(1), reverse=True)
names_top_2 = (name_sorted_desc[0][0], name_sorted_desc[1][0])

two_restaurants = ffl_data[np.logical_or(ffl_data['name'] == names_top_2[0], ffl_data['name'] == names_top_2[1])]




#Standardize longitude and latitude
coordinates = ['latitude', 'longitude']
x = two_restaurants.loc[:, coordinates].values
y = two_restaurants.loc[:, ['name']].values
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
x_df = pd.DataFrame(data= x, columns=coordinates)
y_df = pd.DataFrame(data=y, columns=['name'])
formatted_data = pd.concat([x_df, y_df], axis=1)
# print(formatted_data)

#Graph data with 2 different colors
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(x_df.loc[y_df['name'] == names_top_2[0], 'longitude'],
           x_df.loc[y_df['name'] == names_top_2[0], 'latitude'],
           c = 'r',
           s = 50,
           alpha=.5)
ax.scatter(x_df.loc[y_df['name'] == names_top_2[1], 'longitude'],
           x_df.loc[y_df['name'] == names_top_2[1], 'latitude'],
           c = 'g',
           s = 50,
           alpha=.5)
ax.grid()
plt.show()

# print(x)

#Split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(formatted_data[coordinates].values,
                 formatted_data[['name']].values,
                 train_size=0.8,
                 random_state=91)

# x_train = x_train.reshape(-1, 1)
# x_test = x_test.reshape(-1, 1)
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# print(len(x_train))
# print(len(y_train))
# print(len(x_test))q
# print(x_train)

def myawesomefunction(array_of_distances):
    # print(array_of_distances)
    # print(len(array_of_distances))
    # print(len(array_of_distances[0]))
    print("\n\n\n\n")

    std = np.std(array_of_distances)
    average = np.average(array_of_distances)

    new_array = np.empty(0)
    for value in array_of_distances:
        new_array = np.append(new_array, [average + ((value - average) / std) ** 2 * (value - average)])
    # print(new_array)
    #TODO implement this weighting function
    return np.array(new_array)

#Scoring using KNN
from sklearn.neighbors import KNeighborsClassifier
scores = {}
for neighbor_itr in range(1,100):
    # print(neighbor_itr)
    knn = KNeighborsClassifier(n_neighbors=neighbor_itr)#, weights=myawesomefunction)
    knn.fit(x_train, y_train.ravel())
    pred = knn.predict(x_test)
    score_v1 = accuracy_score(y_test.ravel(), pred)
    scores[neighbor_itr] = score_v1


scores_sorted_desc = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
print(scores_sorted_desc[0])


#TODO Question: how do you get a -0.9 score with categorical data?
#TODO Question: how to balance knn when so many more examples of one category vs another

#TODO check correlation of longitude and latitude
