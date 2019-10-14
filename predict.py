import pandas as pd
import joblib
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler

# Preprocessing
print('Preprocessing...')
conamEncoder = joblib.load('checkpoint/conam-encoder.pkl')
categoryEncoders = [
    ('BE', joblib.load('checkpoint/BE-encoder.pkl')),
    # ('TE', joblib.load('checkpoint/TE-encoder.pkl'))
]

data = pd.read_csv('dataset/test.csv')
output = pd.DataFrame(data['txkey'], columns=['txkey'])
conam = conamEncoder.transform(data['conam'].values.reshape(-1, 1))
data = data.drop(columns=['txkey', 'conam'])

for name, category_enc in categoryEncoders:
    encoded_X = category_enc.transform(data)
    encoded_X['conam_Z'] = conam

    # Load checkpoint
    print(f'{name}-RF Model loading...')
    model = joblib.load(f'checkpoint/{name}-RF.pkl')

    # Predict
    print('Predicting...')
    Y = model.predict(encoded_X)
    output['fraud_ind'] = Y
    output.to_csv(f'submission-{name}.csv', index=False)
