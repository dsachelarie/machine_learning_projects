import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score

"""
Implementation of soft-margin SVM with a linear kernel by stochastic gradient descent.
The hyperparameter C is selected by nested cross-validation.
"""


class SVM:
    """
    Soft-margin SVM with a linear kernel by stochastic gradient descent.
    """

    def __init__(self, C, eta=0.1, iterations=200):
        self.C = C  # penalty coefficient
        self.eta = eta  # step size
        self.iterations = iterations  # number of iterations
        self.w = None  # weight parameters

        return

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)

        for iter in range(self.iterations):
            for i in range(m):
                xi = X[i]
                yi = y[i]

                if yi * np.dot(self.w, xi) < 1:
                    gradient = -yi * xi + self.w / self.C
                else:
                    gradient = self.w / self.C

                self.w = self.w - self.eta * gradient

        return

    def predict(self, X):
        m, n = X.shape
        y = []

        for i in range(m):
            y.append(np.sign(np.dot(self.w, X[i])))

        return y


# Load the data.
X, y = load_breast_cancer(return_X_y=True)
y = 2 * y - 1  # convert the {0, 1} output to {-1, +1}
X /= np.outer(np.ones(X.shape[0]), np.max(np.abs(X), 0))  # normalize the input variables

# Split the data into 5-folds and calculate the f1 score of the own SVM implementation and of sklearn's SVC for each split.
outer_folds = KFold(n_splits=5, random_state=None, shuffle=False)
own_f1s_per_fold = []
svc_f1s_per_fold = []
Cs = [100, 200, 500, 1000, 2000]

for index_train, index_test in outer_folds.split(X):
    Xtrain = X[index_train]
    ytrain = y[index_train]
    Xtest = X[index_test]
    ytest = y[index_test]

    own_f1s_per_c = []
    svc_f1s_per_c = []

    # Get f1 scores for each value of C.
    for i in range(len(Cs)):
        C = Cs[i]

        # Initialize the learners
        svm = SVM(C=C)
        svc = SVC(C=C, kernel='linear')

        # Split the training set into 4-folds.
        inner_folds = KFold(n_splits=4, random_state=None, shuffle=False)
        own_f1s_per_fold_validation = []
        svc_f1s_per_fold_validation = []

        for index_in_train, index_in_test in inner_folds.split(Xtrain):
            Xtrain_in = Xtrain[index_in_train]
            ytrain_in = ytrain[index_in_train]
            Xtest_in = Xtrain[index_in_test]
            ytest_in = ytrain[index_in_test]

            # Run own SVM.
            svm.fit(Xtrain_in, ytrain_in)
            prediction = svm.predict(Xtest_in)
            own_f1s_per_fold_validation.append(f1_score(ytest_in, prediction))

            # Run sklearn's SVC.
            svc.fit(Xtrain_in, ytrain_in)
            prediction = svc.predict(Xtest_in)
            svc_f1s_per_fold_validation.append(f1_score(ytest_in, prediction))

        # Compute the mean f1 score for a given C value.
        own_f1s_per_c.append(np.mean(own_f1s_per_fold_validation))
        svc_f1s_per_c.append(np.mean(svc_f1s_per_fold_validation))

    # Select the best C value with the highest mean F1 score for each method.
    C_own = Cs[np.argmax(own_f1s_per_c)]
    C_svc = Cs[np.argmax(svc_f1s_per_c)]

    # Run own SVM and sklearn's SVC with the best C values.
    svm = SVM(C=C_own)
    svm.fit(Xtrain, ytrain)
    prediction = svm.predict(Xtest)
    own_f1s_per_fold.append(f1_score(ytest, prediction))

    svc = SVC(C=C_svc, kernel='linear')
    svc.fit(Xtrain, ytrain)
    prediction = svc.predict(Xtest)
    svc_f1s_per_fold.append(f1_score(ytest, prediction))

print("Own SVM mean f1 score: " + str(np.mean(own_f1s_per_fold)))
print("sklearn SVC mean f1 score: " + str(np.mean(svc_f1s_per_fold)))
