import sys
import numpy as np
import random
from sklearn.linear_model import Perceptron

"""
Calculation of generalization bounds (a classifier's maximum error on a dataset) based on VC-dimension and Rademacher complexity. 
The data distribution is only taken into account by Rademacher, VC-dimension is less informative though easier to compute.
"""


def calculate_empirical_risk(classifier, x_train, y_train):
    predictions = classifier.predict(x_train)
    risk = 0

    for k in range(len(ytr)):
        if predictions[k] != y_train[k]:
            risk += 1

    risk /= len(ytr)

    return risk


# Read dataset.
Xtr = np.load("Xtr.npy")
Xtst = np.load("Xtst.npy")
ytr = np.load("ytr.npy")
ytst = np.load("ytst.npy")
m = len(ytr)

# Train classifier.
bc = Perceptron()
bc.fit(Xtr, ytr)
preds = bc.predict(Xtst)

# Calculate empirical risk and Rademacher complexity.
empirical_risk = calculate_empirical_risk(bc, Xtr, ytr)
no_labelings = 1000
lowest_hypothesis_value = sys.maxsize

for i in range(no_labelings):
    hypothesis = []

    for j in range(len(ytr)):
        hypothesis.append(random.choice([0, 1]))

    hypothesis_risk = calculate_empirical_risk(bc, Xtr, hypothesis)

    if hypothesis_risk < lowest_hypothesis_value:
        lowest_hypothesis_value = hypothesis_risk

rademacher_complexity = 1 / 2 - lowest_hypothesis_value

# Calculate generalization bounds.
delta = 0.05
generalization_bound_vc = empirical_risk + np.sqrt(6 * np.log(np.e * m / 3) / m) + np.sqrt(np.log(1 / delta) / 2 / m)
generalization_bound_rademacher = empirical_risk + rademacher_complexity + 3 * np.sqrt(np.log10(2 / delta) / 2 / m)

print("VC-dimension generalization bound: " + str(generalization_bound_vc))
print("Rademacher complexity generalization bound: " + str(generalization_bound_rademacher))
