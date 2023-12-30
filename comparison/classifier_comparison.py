import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

"""
Accuracy comparison for several classifiers. 5-fold cross validation performed for hyperparameter tuning.
"""


def get_model(classifier, param):
    if classifier == "linear_svc_crammer_singer":
        return LinearSVC(C=param, max_iter=10000, multi_class="crammer_singer", dual="auto")

    elif classifier == "linear_svc_ovo":
        return SVC(C=param, max_iter=10000, kernel="linear", decision_function_shape="ovo")

    elif classifier == "linear_svc_ova":
        return LinearSVC(C=param, max_iter=10000, multi_class="ovr", dual="auto")

    elif classifier == "decision_tree":
        return DecisionTreeClassifier(max_depth=param, random_state=0)

    elif classifier == "random_forest":
        return RandomForestClassifier(n_estimators=param, random_state=0)

    elif classifier == "mlp_classifier":
        return MLPClassifier(alpha=param, max_iter=10000, random_state=0)

    elif classifier == "logistic_regression":
        return LogisticRegression(C=param, multi_class="multinomial", max_iter=10000)

    else:
        raise Exception("Unknown model!")

def train_test(classifier, params, Xtr, ytr, Xtst, ytst):
    k_fold = KFold()
    accuracies = []

    for param in params:
        accuracies_per_fold = []

        for (train_i, validate_i) in k_fold.split(Xtr):
            train_X = [Xtr[i] for i in train_i]
            train_y = [ytr[i] for i in train_i]
            validate_X = [Xtr[i] for i in validate_i]
            validate_y = [ytr[i] for i in validate_i]

            model = get_model(classifier, param)
            model.fit(train_X, train_y)
            prediction = model.predict(validate_X)
            accuracies_per_fold.append(accuracy_score(validate_y, prediction))

        accuracies.append(np.mean(accuracies_per_fold))

    param = params[np.argmax(accuracies)]
    model = get_model(classifier, param)
    model.fit(Xtr, ytr)
    predictions = model.predict(Xtst)
    score = accuracy_score(ytst, predictions)
    print(f"{classifier} accuracy: {score}")

    return model

Xtr = np.load("Xtr.npy")
Xtst = np.load("Xtst.npy")
ytr = np.load("ytr.npy")
ytst = np.load("ytst.npy")

linear_svc_crammer_singer_param = [1e-3, 1e-2, 1e-1, 1, 10, 100]
linear_svc_ovo_param = [1e-3, 1e-2, 1e-1, 1, 10, 100]
linear_svc_ova_param = [1e-3, 1e-2, 1e-1, 1, 10, 100]
decision_tree_param = [1, 2, 3, 4, 5, 6, 7, 8, 9]
random_forest_param = [25, 50, 75, 100]
mlp_classifier_param = [1e-3, 1e-2, 1e-1, 1, 10, 100]
logistic_regression_param = [1e-3, 1e-2, 1e-1, 1, 10, 100]

train_test("linear_svc_crammer_singer", linear_svc_crammer_singer_param, Xtr, ytr, Xtst, ytst)
train_test("linear_svc_ovo", linear_svc_ovo_param, Xtr, ytr, Xtst, ytst)
train_test("linear_svc_ova", linear_svc_ova_param, Xtr, ytr, Xtst, ytst)
train_test("decision_tree", decision_tree_param, Xtr, ytr, Xtst, ytst)
train_test("random_forest", random_forest_param, Xtr, ytr, Xtst, ytst)
train_test("mlp_classifier", mlp_classifier_param, Xtr, ytr, Xtst, ytst)
train_test("logistic_regression", logistic_regression_param, Xtr, ytr, Xtst, ytst)
