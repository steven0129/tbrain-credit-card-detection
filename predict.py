import pandas as pd
import joblib
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from types import MethodType
from numpy import hstack

# Load model
print('Loading model...')
model = joblib.load('checkpoint/gbdt-model.pkl')
encoder = joblib.load('checkpoint/encoder.pkl')
def testing_set(self):
    train_set = pd.read_csv(encoder.train_set)
    test_set = pd.read_csv(encoder.test_set)
    
    Y = train_set['fraud_ind']
    X = train_set.drop(columns=['fraud_ind']).append(test_set, ignore_index=True)
    conam = encoder.conam_enc.fit_transform(X['conam'].values.reshape(-1, 1)).astype('float16')
    txkey = test_set[['txkey']]
    X = X.drop(columns=['txkey', 'locdt', 'conam'])
    X = X.astype(str)
    keys = list(X.keys())

    for key in keys:
        X[key] = encoder.label_encs[key].fit_transform(X[key].values)
        print(f'label_encs.{key}.classes_.shape = {encoder.label_encs[key].classes_.shape}')

    X = encoder._hash_func(X)
    X = encoder.onehot_enc.transform(X)
    X = encoder.svd.transform(X)
    X = hstack((X, conam))
    X = X[train_set.shape[0]:, :]

    return X, txkey

encoder.testing_set = MethodType(testing_set, model)

# Preprocessing
print('Preprocessing...')
X, txkey = encoder.testing_set()
print(f'X.shape={X.shape}')

# Predict
print('Predicting...')
output = txkey
Y = model.predict(X)
output['fraud_ind'] = Y.astype(int)
output.to_csv('submission.csv', index=False)