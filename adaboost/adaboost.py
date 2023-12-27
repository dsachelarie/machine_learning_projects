import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.datasets import make_circles, make_moons
from sklearn.neural_network import MLPClassifier

"""
Implementation of AdaBoost.
"""


class AdaBoost:
    def __init__(self, learners):
        self.learners = learners  # list of classifiers
        self.T = len(learners)  # number of learners
        self.alpha = None  # the weights of the learners
        self.D = None  # the weights of the data examples

    def fit(self, X, y):
        m, n = X.shape

        self.D = np.ones(m) / m
        self.alpha = np.zeros(self.T)

        for t in range(self.T):
            self.learners[t].fit(X, y)
            h_predictions = self.learners[t].predict(X)
            e = 0

            for i in range(m):
                e += (h_predictions[i] != y[i]) * self.D[i]

            self.alpha[t] = 1 / 2 * np.log((1 - e) / e)
            z = 2 * np.sqrt(e * (1 - e))

            for i in range(m):
                self.D[i] = self.D[i] * np.exp(-self.alpha[t] * y[i] * h_predictions[i]) / z

    def predict(self, Xtest):
        y_predict = np.zeros(len(Xtest))

        for t in range(self.T):
            t_prediction = self.learners[t].predict(Xtest)

            for i in range(len(Xtest)):
                if self.alpha[t] > 0:
                    y_predict[i] += self.alpha[t] * t_prediction[i]

        return np.sign(y_predict)

    def get_alphas(self):
        return self.alpha


datasets = [
    make_circles(n_samples=1000, noise=0.3, factor=0.5, random_state=0),
    make_moons(n_samples=1000, noise=0.3, random_state=0)
]

# Construct the list of learners.
classifiers = []
i = 10
while i <= 100:
    classifiers.append(
        make_pipeline(
            StandardScaler(),
            MLPClassifier(
                solver="adam",
                alpha=0.0001,
                random_state=0,
                max_iter=1000,
                early_stopping=False,
                hidden_layer_sizes=[i]),
        )
    )

    i += 10

# Run AdaBoost on the two datasets.
for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    y = 2 * y - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    ada_boost = AdaBoost(classifiers)
    ada_boost.fit(X_train, y_train)
    predictions = ada_boost.predict(X_test)

    print(f"alphas dataset {ds_cnt}: {ada_boost.get_alphas()}")
    print(f"f1 score dataset {ds_cnt}: {f1_score(predictions, y_test)}")
