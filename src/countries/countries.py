import pandas as pd
import numpy as np
import math
import operator


columns_to_keep = [
    'Country',
    'GDP ($ per capita)',
    'Literacy (%)',
    'Region'
]
data = pd.read_csv('3-countries of the world-small-large_countries_only.csv')
data = data[columns_to_keep]

print(data.head())
print(data)



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

    #### Start of STEP 3.3
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    #### End of STEP 3.3
    classVotes = {}

    #### Start of STEP 3.4
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = training_set.iloc[neighbors[x]][-1]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #### End of STEP 3.4

    #### Start of STEP 3.5
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return (sortedVotes[0][0], neighbors)
    #### End of STEP 3.5



# Creating a dummy testset
testSet = [[5000, 85]]
test = pd.DataFrame(testSet)