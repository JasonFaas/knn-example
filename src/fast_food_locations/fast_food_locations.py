import pandas as pd
import operator
import numpy as np

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
province_dict = {}
for ffl_lab, ffl_row in ffl_data.iterrows():
    name_dict[ffl_row['name']] = name_dict.get(ffl_row['name'], 0) + 1
    province_dict[ffl_row['province']] = province_dict.get(ffl_row['province'], 0) + 1
    
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

#TODO do scoring using linear regression
#TODO use knn
#TODO check correlation of longitude and latitude
