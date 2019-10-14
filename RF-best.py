import numpy as np
import pandas as pd
import joblib
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from custom_model import FraudDetectClassifier

# Preprocessing
print('Preprocessing...')
data = pd.read_csv('dataset/train.csv')
data = shuffle(data, random_state=42)
Y = data['fraud_ind']
conamEncoder = MinMaxScaler()
conam = conamEncoder.fit_transform(data['conam'].values.reshape(-1, 1))
data = data.drop(columns=['txkey', 'fraud_ind', 'conam'])
categoryEncoders = [
    ('BE', ce.BinaryEncoder(cols=list(data.keys()))),
    # ('TE', ce.TargetEncoder(cols=list(data.keys())))
]

model = FraudDetectClassifier()

for name, category_enc in categoryEncoders:
    encoded_X = category_enc.fit_transform(data)
    encoded_X['conam_Z'] = conam
    model.fit(encoded_X, Y)
    print('Saving model...')
    joblib.dump(model, f'checkpoint/{name}-RF.pkl')

    # lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(encoded_X, fraud_ind)
    # selector = SelectFromModel(lsvc, prefit=True)
    # X = selector.transform(encoded_X)
    # Y = fraud_ind
    # print(f'X.shape = {X.shape}')

    # # Train model
    # print(f'Training {name}-RF model...')
    # model = RandomForestClassifier(
    #     n_estimators=63,
    #     verbose=1,
    #     n_jobs=12,
    #     oob_score=True,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_features=None,
    #     max_depth=104
    # )
    
    # model.fit(X, Y)
    # oob_f1_score = f1_score(Y, np.argmax(model.oob_decision_function_, axis=1))
    # oob_confuse_matrix = pd.DataFrame(
    #     confusion_matrix(
    #         Y,
    #         np.argmax(model.oob_decision_function_, axis=1)
    #     ),
    #     index = ['N', 'Y'], columns=['N', 'Y']
    # ) / X.shape[0]

    # print(f'{name}-RF oob_f1_score = {oob_f1_score}')
    # FILE.write(f'{name}-RF oob_f1_score = {oob_f1_score}\n')
    # plt.figure()
    # heatmap = sn.heatmap(oob_confuse_matrix, annot=True, cmap="YlGnBu")
    # plt.savefig(f'checkpoint/{name}-confusion-matrix.png')

    # # Save model
    # print('Saving model...')
    # joblib.dump(model, f'checkpoint/{name}-RF.pkl')
    # joblib.dump(category_enc, f'checkpoint/{name}-encoder.pkl')
    # joblib.dump(selector, f'checkpoint/{name}-selector.pkl')

joblib.dump(conamEncoder, 'checkpoint/conam-encoder.pkl')
