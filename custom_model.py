import numpy as np
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel

class FraudDetectClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=0.1, n_estimators=63, min_samples_split=2, min_samples_leaf=1, max_features=None, max_depth=104):
        lsvc = LinearSVC(C=C, penalty='l1', dual=False)
        self.selector = SelectFromModel(lsvc)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            verbose=1,
            n_jobs=-1,
            oob_score=True,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_depth=max_depth
        )

    def fit(self, X, Y):
        X, Y = check_X_y(X, Y)
        self.classes_ = unique_labels(Y)
        self.X_ = X
        self.y_ = Y

        print(f'X.shape={X.shape}')
        print('Feature Selection...')
        X = self.selector.fit_transform(X, Y)
        print(f'X.shape={X.shape}')
        print('Training Model...')
        self.model.fit(X, Y)
        oob_f1_score = f1_score(Y, np.argmax(self.model.oob_decision_function_, axis=1))
        print(f'oob_f1_score = {oob_f1_score}')

        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        X = self.selector.transform(X)
        return self.model.predict(X)
