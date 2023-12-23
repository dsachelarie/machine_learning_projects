import numpy as np
from sklearn.svm import LinearSVC

"""
Implementation of a linear regression model and comparison to SVM.
"""


def get_accuracy_recall(pred, test):
    no_correct = 0
    true_positive = 0
    false_negative = 0

    for (a, b) in zip(pred, test):
        if a == b:
            no_correct += 1

            if b == 1:
                true_positive += 1

        elif b == 1:
            false_negative += 1

    return no_correct / len(test), true_positive / (true_positive + false_negative)


# Get train and test sets.
X = np.load("X.npy")
y = np.load("y.npy")
np.random.seed(0)
order = np.random.permutation(len(y))
tr_samples = order[:int(0.5 * len(y))]
tst_samples = order[int(0.5 * len(y)):]
Xtr = X[tr_samples, :]
Xtst = X[tst_samples, :]
ytr = y[tr_samples]
ytst = y[tst_samples]

# Train and get predictions from SVM.
svm = LinearSVC(dual=False)
svm.fit(Xtr, ytr)
predictions1 = svm.predict(Xtst)

# Train linear regression model.
w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Xtr), Xtr)), np.transpose(Xtr)), ytr)

# Get regression predictions.
predictions2 = []

for sample in Xtst:
    predictions2.append(np.matmul(w, sample))

# Get binary labels from regression predictions.
for i in range(len(predictions2)):
    predictions2[i] = predictions2[i] >= 0.5

# Get accuracy and recall for the models' predictions.
accuracy_svm, recall_svm = get_accuracy_recall(predictions1, ytst)
accuracy_lr, recall_lr = get_accuracy_recall(predictions2, ytst)

print("Accuracy svm: " + str(accuracy_svm) + "\nAccuracy lr: " + str(accuracy_lr))
print("Recall svm: " + str(recall_svm) + "\nRecall lr: " + str(recall_lr))
