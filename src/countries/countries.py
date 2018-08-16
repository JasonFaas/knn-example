import pandas as pd
import numpy as np
import math
import operator





# Defining a function which calculates euclidean distance between two data points
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)


# Defining our KNN model
def knn(training_set, test_instance, k):
    distances = {}
    sort = {}

    length = test_instance.shape[1]

    #### Start of STEP 3
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(training_set)):
        #### Start of STEP 3.1
        dist = euclideanDistance(test_instance, training_set.iloc[x], length)

        distances[x] = dist[0]
        #### End of STEP 3.1

    #### Start of STEP 3.2
    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    #### End of STEP 3.2

    neighbors = []
    neighbors_names = []

    #### Start of STEP 3.3
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    #### End of STEP 3.3
    classVotes = {}

    print(neighbors)

    #### Start of STEP 3.4
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = training_set.iloc[neighbors[x]][-1]
        neighbors_names.append([training_set.iloc[neighbors[x]].name, training_set.iloc[neighbors[x]][-1]])

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #### End of STEP 3.4

    #### Start of STEP 3.5
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return (sortedVotes[0][0], neighbors, neighbors_names)
    #### End of STEP 3.5



# Creating a dummy testset
testSet = [[50000, 85.5]]
test = pd.DataFrame(testSet)



columns_to_keep = [
    'GDP ($ per capita)',
    'Literacy (%)',
    'Region'
]
data = pd.read_csv('5-countries of the world-A-H.csv',index_col=0)
data = data[columns_to_keep]
# print(data.head())
# print(data.loc['China'].name)
# print(data)

# print(test.shape[1])
# print(test[0])
# print(test[1])
# print(data.iloc[0][0])
# print(data.iloc[0][1])
# print(data.iloc[2])
# print(data.iloc[3])


#### Start of STEP 2
# Setting number of neighbors = 1
k = 7
#### End of STEP 2
# Running KNN model
result,neigh,neigh_names = knn(data, test, k)

# Predicted class
print(result)
print(neigh)
print(neigh_names)


