import numpy as np
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.metrics import r2_score, balanced_accuracy_score, accuracy_score
from tabulate import tabulate
import matplotlib.pyplot as plt
from time import sleep


encodings = ["STML", "IGTD", "DI", "RETIRE"]
meta = np.load("results/meta_dataset.npy")
meta_names = meta[:, 0]
X = meta[:, [6, 23]].astype(float)
y = meta[:, 24].astype(float)

dataset_indx = np.arange(0, 80, 4)

# print(X[dataset_indx, 0])
# exit()

instances_idxs = np.arange(0, X.shape[0], 1)
scores = []
preds = []
pred_best_clf = []
best_clf = []
for fold_id in range(20):
    test = np.arange(dataset_indx[fold_id], dataset_indx[fold_id]+4, 1)
    train = np.delete(instances_idxs, test)
    
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    
    reg = XGBRegressor(n_estimators =100).fit(X, y)
    pred = reg.predict(X_test)
    pred_best_clf.append(np.argmax(pred))
    best_clf.append(np.argmax(y_test))
    preds.append(pred)
    score = r2_score(y_test, pred)
    scores.append(score)
    print(pred)

scores = np.array(scores)
preds = np.array(preds)
pred_best_clf = np.array(pred_best_clf)
best_clf = np.array(best_clf)



t = tabulate(np.concatenate((meta[:, 0].reshape(-1, 1), preds.reshape(-1, 1), y.reshape(-1, 1)), axis=1), tablefmt="simple")
print(t)
print(np.mean(scores))
print(best_clf)
print(pred_best_clf)
print(accuracy_score(best_clf, pred_best_clf))



exit()
"""
complexity x bac plotfor each encoding
"""
for j in range(1, 22, 1):
    print(j)
    stml_idx = np.arange(0, 80, 4)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt_data = meta[:, [j, 24]].astype(float)

    for encoding_id, encoding in enumerate(encodings):
        indices = stml_idx + encoding_id
        encoding_complexity = plt_data[indices, 0]
        encoding_bac = plt_data[indices, 1]
    
        sorted = np.argsort(encoding_complexity)
    
        ax.plot(encoding_complexity[sorted], encoding_bac[sorted], label=encoding)

    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/complexity/relation.png")
    sleep(1)
    plt.close()