import numpy as np

test_array = np.array([3.45, 9.99, 0.22, 1, 2, 3, 4, 5, 6, 7, 8])
std = np.std(test_array)
print(std)
average = np.average(test_array)
print(average)
print(np.median(test_array))

new_array = np.empty(0)
for value in test_array:
    new_array = np.append(new_array, average + ((value - average) / std) ** 2 * (value - average))


print(new_array)

