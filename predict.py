import pandas as pd
import joblib
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# Load model
print('Loading model...')
model = joblib.load('checkpoint/gbdt-model.pkl')
encoder = joblib.load('checkpoint/encoder.pkl')

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