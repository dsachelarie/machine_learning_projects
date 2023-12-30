import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""
Comparison of various feature transformation techniques, when variable ranking is performed and top-k variables (k ranging from 1 to 30) are selected.
"""


def get_ranking_from_coefficients(coefficients: []) -> []:
    ranking = np.zeros(30)
    remaining = 30

    while remaining > 0:
        max_pos = np.argmax(coefficients)
        ranking[max_pos] = 30 - remaining
        coefficients[max_pos] = -1
        remaining -= 1

    return ranking


X_original = np.load("X.npy")
y_original = np.load("y.npy")
tr = np.load("tr.npy")
tst = np.load("tst.npy")

# Transform features using centering.
data_centering = np.copy(X_original)

for i in range(30):
    data_centering[:, i] -= np.mean(data_centering[:, i])

# Transform features using standardization.
data_standardization = np.copy(data_centering)

for i in range(30):
    data_standardization[:, i] /= np.sqrt(np.var(data_standardization[:, i]))

# Transform features using unit range.
data_unit_range = np.copy(X_original)

for i in range(30):
    data_unit_range[:, i] = (data_unit_range[:, i] - np.min(data_unit_range[:, i])) / (np.max(data_unit_range[:, i]) - np.min(data_unit_range[:, i]))

# Transform features using normalization with L2 norm.
data_normalization = np.copy(X_original)

for i in range(569):
    data_normalization[i, :] /= np.linalg.norm(data_normalization[i, :])

# Perform variable ranking using squared Pearson correlation.
data_list = [X_original, data_centering, data_standardization, data_unit_range, data_normalization]
ranking = [None, None, None, None, None]
mean_y = np.mean(y_original)

for dataset_id in range(5):
    coefficients = np.zeros(30)

    for j in range(30):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        mean_x = np.mean(data_list[dataset_id][:, j])

        for i in range(569):
            sum1 += (data_list[dataset_id][i, j] - mean_x) * (y_original[i] - mean_y)
            sum2 += (data_list[dataset_id][i, j] - mean_x)**2
            sum3 += (y_original[i] - mean_y)**2

        coefficients[j] = sum1**2 / sum2 / sum3

    ranking[dataset_id] = np.copy(get_ranking_from_coefficients(coefficients))

# Train and test SVC for each feature transformation, using 1-30 highest-ranking variables.
results = [None, None, None, None, None]

for dataset_id in range(5):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in tr:
        X_train.append(data_list[dataset_id][i, ])
        y_train.append(y_original[i])

    for i in tst:
        X_test.append(data_list[dataset_id][i, ])
        y_test.append(y_original[i])

    dataset_ranking = ranking[dataset_id]
    results_dataset = np.zeros(30)

    for k in range(30, 0, -1):
        svc = SVC()
        svc.fit(X_train, y_train)
        results_dataset[k - 1] = accuracy_score(y_test, svc.predict(X_test))
        remove_id = np.where(dataset_ranking == k - 1)[0][0]
        dataset_ranking = np.delete(dataset_ranking, remove_id, 0)
        X_train = np.delete(X_train, remove_id, 1)
        X_test = np.delete(X_test, remove_id, 1)

    results[dataset_id] = results_dataset

plt.plot(results[0], label="None", color='yellow')
plt.plot(results[1], label="Centering", color='r')
plt.plot(results[2], label="Standardization", color='b')
plt.plot(results[3], label="Unit Range", color='k')
plt.plot(results[4], label="Normalization", color='orange')
plt.legend()
plt.show()
