import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.model_selection import KFold

"""
Analysis of the robustness of two models, SVC and Lasso, when noise is added to the data.
"""

X = np.load("X.npy")
y = np.load("y.npy")
X_noise = np.load("X_noise.npy")
y_noise = np.random.choice([-1, 1], size=30)

params_lasso = [1, 1e-1, 1e-2, 1e-3, 1e-4]
params_rbf = [1e-3, 1e-2, 1e-1, 1, 10, 100]
params_svm = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
lasso_accuracies_per_added_noise = []
svm_accuracies_per_added_noise = []

# Calculate accuracy for a selected number of added noisy features.
for number_of_noisy_features in range(31):
    new_X = []

    # Add noisy features to the dataset.
    for i in range(len(X)):
        if number_of_noisy_features == 0:
            new_X.append(X[i])

        else:
            new_X.append(np.concatenate((X[i], X_noise[i][:number_of_noisy_features])))

    kf_outer = KFold(n_splits=5)
    lasso_accuracies = []
    svm_accuracies = []

    for (trn_i, test_i) in kf_outer.split(new_X):
        kf_inner = KFold(n_splits=2)
        trn_X = [new_X[i] for i in trn_i]
        trn_y = [y[i] for i in trn_i]
        test_X = [new_X[i] for i in test_i]
        test_y = [y[i] for i in test_i]
        lasso_scores = []
        svm_scores = np.zeros((7, 6))

        # Get accuracy scores for Lasso for each hyperparameter.
        for param in params_lasso:
            lasso_val_scores = []

            for (train_i, validate_i) in kf_inner.split(trn_X):
                train_X = [new_X[i] for i in train_i]
                train_y = [y[i] for i in train_i]
                validate_X = [new_X[i] for i in validate_i]
                validate_y = [y[i] for i in validate_i]

                lasso = Lasso(param)
                lasso.fit(train_X, train_y)
                predictions = np.sign(lasso.predict(validate_X))
                
                lasso_val_scores.append(accuracy_score(validate_y, predictions))

            lasso_scores.append(np.mean(lasso_val_scores))

        # Get accuracy scores for SVC for each combination of hyperparameters.
        for i, param1 in enumerate(params_svm):
            for j, param2 in enumerate(params_rbf):
                svm_val_scores = []

                for (train_i, validate_i) in kf_inner.split(trn_X):
                    train_X = [new_X[i] for i in train_i]
                    train_y = [y[i] for i in train_i]
                    validate_X = [new_X[i] for i in validate_i]
                    validate_y = [y[i] for i in validate_i]

                    svm = SVC(C=param1, gamma=param2)
                    svm.fit(train_X, train_y)
                    predictions = svm.predict(validate_X)

                    svm_val_scores.append(accuracy_score(validate_y, predictions))

                svm_scores[i][j] = np.mean(svm_val_scores)

        lasso = Lasso(params_lasso[np.argmax(lasso_scores)])
        lasso.fit(trn_X, trn_y)
        predictions = np.sign(lasso.predict(test_X))
        lasso_accuracies.append(accuracy_score(test_y, predictions))

        svm_indices = np.unravel_index(np.argmax(svm_scores), svm_scores.shape)
        svm = SVC(C=params_svm[svm_indices[0]], gamma=params_rbf[svm_indices[1]])
        svm.fit(trn_X, trn_y)
        predictions = svm.predict(test_X)
        svm_accuracies.append(accuracy_score(test_y, predictions))

    lasso_accuracies_per_added_noise.append(np.mean(lasso_accuracies))
    svm_accuracies_per_added_noise.append(np.mean(svm_accuracies))

print(f"Lasso accuracy for 0-30 added noisy features: {np.round(lasso_accuracies_per_added_noise, decimals=2)}")
print(f"SVC accuracy for 0-30 added noisy features: {np.round(svm_accuracies_per_added_noise, decimals=2)}")
