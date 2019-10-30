from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

def gt(a, b):
    if a[1] > b[1]:
        return a
    else:
        return b

def kfold_val(model, X, Y, cv=10):
    score = 0
    for train_idx, val_idx in KFold(n_splits=cv).split(X):
        x_train = X[train_idx]
        x_val = X[val_idx]
        y_train = Y[train_idx]
        y_val = Y[val_idx]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        score += f1_score(y_val, y_pred)

    return score / cv