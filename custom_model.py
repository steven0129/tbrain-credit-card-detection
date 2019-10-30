import numpy as np
import pandas as pd
import xgboost
import lightgbm as lgb
import math
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from scipy.misc import derivative

class RFDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, K_best=200, n_estimators=63, min_samples_split=2, min_samples_leaf=1, max_features=None, max_depth=104):
        self.anova = SelectKBest(f_classif, k=K_best)
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
        X = self.anova.fit_transform(X, Y)
        print(f'X.shape={X.shape}')
        print('Training Model...')
        self.model.fit(X, Y)
        oob_f1_score = f1_score(Y, np.argmax(self.model.oob_decision_function_, axis=1))
        print(f'oob_f1_score = {oob_f1_score}')

        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        X = self.anova.transform(X)
        return self.model.predict(X)

class XGBDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=5000, max_depth=5, min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1):
        self.xgb = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            n_jobs=48,
            random_state=0
        )
    
    def fit(self, X, Y):
        X, Y = check_X_y(X, Y)
        self.classes_ = unique_labels(Y)
        self.X_ = X
        self.y_ = Y
        self.xgb.fit(X, Y)
        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        return self.xgb.predict(X)

class LightGBDT(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=5000, lr=0.1, base_lr=0.01, focal_alpha=1, focal_gamma=2):
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.n_estimators = n_estimators
        self.base_lr = base_lr
        self.lr = lr
        self.gbdt = LGBMClassifier(
            boosting_type='gbdt',
            n_estimators=n_estimators,
            learning_rate=lr,
            objective=self._partial_focal,
            random_state=0,
            tree_learner_type='data_parallel',
            silent=-1,
            verbose=-1
        )
    
    def fit(self, X, Y, eval_set=[]):
        self.classes_ = unique_labels(Y)
        self.X_ = X
        self.y_ = Y
        self.gbdt.fit(X, Y,
            eval_set=eval_set,
            eval_metric=lambda y_true, y_pred: [
                self._focal_eval(y_true, y_pred),
                self._f1_score(y_true, y_pred)
            ],
            verbose=100,
            early_stopping_rounds=5000,
            callbacks=[lgb.reset_parameter(learning_rate=self._learning_rate_decay)]
        )

        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        X = self.gbdt.predict(X, raw_score=True)
        X = self._sigmoid(X)
        pred_labels = np.around(X).astype(int)
        return pred_labels

    def _f1_score(self, y_true, y_pred):
        y_pred = self._sigmoid(y_pred)
        pred_labels = np.around(y_pred)
        return 'f1', f1_score(y_true, pred_labels), True

    def _focal_eval(self, y_true, y_pred):
        focal_avg = np.average(self._focal_loss(y_pred, y_true))
        return 'focal_loss', focal_avg, False

    def _partial_focal(self, y_true, y_pred):
        grad = derivative(lambda x: self._focal_loss(x, y_true), y_pred, n=1, dx=1e-6)
        hess = derivative(lambda x: self._focal_loss(x, y_true), y_pred, n=2, dx=1e-6)
        return grad, hess

    def _focal_loss(self, x, t):
        a = self.focal_alpha
        g = self.focal_gamma
        p = self._sigmoid(x)
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )

    def _sigmoid(self, x):
        x = np.clip(x, -20, 20) # Avoid p or (1-p) floaing overflow
        x = 1 / (1 + np.exp(-x))
        return x

    def _scaled_tanh(self, x):
        return (np.tanh(x) + 1) / 2

    def _learning_rate_decay(self, curr_round):
        total_round = self.n_estimators
        cosine_decayed = 0.5 * (1 + math.cos(math.pi * curr_round / total_round))
        decayed = (1 - self.base_lr) * cosine_decayed + self.base_lr
        decayed_lr = self.lr * decayed
        if curr_round % 100 == 0: print(f'curr_round = {curr_round}, learning rate cosine decayed = {decayed_lr}')
        return decayed_lr

class FocalXGBDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, focal_gamma, n_estimators=5000, max_depth=5, min_child_weight=1, tree_gamma=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1):
        self.n_estimators=n_estimators
        self.max_depth = max_depth
        self.min_child_weight=min_child_weight
        self.tree_gamma=tree_gamma
        self.focal_gamma=focal_gamma
        self.subsample=subsample
        self.colsample_bytree=colsample_bytree
        self.scale_pos_weight=scale_pos_weight

    def fit(self, X, Y):
        X, Y = check_X_y(X, Y)
        self.classes_ = unique_labels(Y)
        self.X_ = X
        self.y_ = Y
        self._booster = xgboost.train(
            params={
                'max_depth': self.max_depth,
                'min_child_weight': self.min_child_weight,
                'gamma': self.tree_gamma,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'scale_pos_weight': self.scale_pos_weight
            }, 
            dtrain=xgboost.DMatrix(X, label=Y, nthread=-1),
            num_boost_round=self.n_estimators,
            obj=self._focal_binary_loss
        )

        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        X = xgboost.DMatrix(X)
        logistic = self._booster.predict(X)
        return np.where(logistic < 0.5, 0, 1)

    def _robust_pow(self, num_base, num_pow):
        # numpy does not permit negative numbers to fractional power
        # use this to perform the power algorithmic

        return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)

    def _focal_binary_loss(self, pred, dtrain):
        gamma_indct = self.focal_gamma
        # retrieve data from dtrain matrix
        label = dtrain.get_label()
        # compute the prediction with sigmoid
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        # gradient
        # complex gradient with different parts
        g1 = sigmoid_pred * (1 - sigmoid_pred)
        g2 = label + ((-1) ** label) * sigmoid_pred
        g3 = sigmoid_pred + label - 1
        g4 = 1 - label - ((-1) ** label) * sigmoid_pred
        g5 = label + ((-1) ** label) * sigmoid_pred
        # combine the gradient
        grad = gamma_indct * g3 * self._robust_pow(g2, gamma_indct) * np.log(g4 + 1e-9) + \
               ((-1) ** label) * self._robust_pow(g5, (gamma_indct + 1))
        # combine the gradient parts to get hessian components
        hess_1 = self._robust_pow(g2, gamma_indct) + \
                 gamma_indct * ((-1) ** label) * g3 * self._robust_pow(g2, (gamma_indct - 1))
        hess_2 = ((-1) ** label) * g3 * self._robust_pow(g2, gamma_indct) / g4
        # get the final 2nd order derivative
        hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma_indct +
                (gamma_indct + 1) * self._robust_pow(g5, gamma_indct)) * g1

        return grad, hess
