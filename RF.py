import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

# Preprocessing
data = pd.read_csv('dataset/train.csv')
fraud_ind = data['fraud_ind']
conam = MinMaxScaler().fit_transform(data['conam'].values.reshape(-1, 1))
data = data.drop(columns=['fraud_ind', 'conam', 'txkey'])
data = BinaryEncoder(cols=list(data.keys())).fit_transform(data)
data['conam_Z'] = conam
data['fraud_ind'] = fraud_ind
# RandomForest
X = data.drop(columns=['fraud_ind'])
Y = data['fraud_ind']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(f'X_train.shape={X_train.shape}, X_test.shape={X_test.shape}, Y_train.shape={Y_train.shape}, Y_test.shape={Y_test.shape}')

# Hyperparameter
# Number of trees in random forest
n_estimators = [i for i in range(10, 200, 1)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2', None]
# Maximum number of levels in tree
max_depth = [x for x in range(10, 110, 1)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

print(f'參數數量: {len(n_estimators) * len(max_features) * len(max_depth) * len(min_samples_split) * len(min_samples_leaf)}')

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'n_jobs': [10],
               'verbose': [1]}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring='roc_auc', n_iter = 50, cv=3, verbose=150, random_state=42)
rf_random.fit(X_train, Y_train)
print(rf_random.best_params_)
with open('best_params.txt', 'w') as FILE:
    FILE.write(str(rf_random.best_params_))
