import pandas as pd
from category_encoders import *
from sklearn.datasets import load_boston
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm
from vis import tsne_vis

# Preprocessing
data = pd.read_csv('dataset/train.csv')
fraud_ind = data['fraud_ind']
conam = MinMaxScaler().fit_transform(data['conam'].values.reshape(-1, 1))
data = data.drop(columns=['fraud_ind', 'conam'])
data = BinaryEncoder(cols=list(data.keys())).fit_transform(data)
data['conam_Z'] = conam
data['fraud_ind'] = fraud_ind

# Data visulization
#tsne_vis(data, verbose=1, num_sample=10000)

# Model selection
X = data.drop(columns=['fraud_ind'])
Y = data['fraud_ind']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(f'X_train.shape={X_train.shape}, X_test.shape={X_test.shape}, Y_train.shape={Y_train.shape}, Y_test.shape={Y_test.shape}')

models = []
models.append(('LR', LogisticRegression()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', LinearSVC()))
models.append(('RF', RandomForestClassifier()))
models.append(('LOF', LocalOutlierFactor()))

results = []
names = []

for name, model in tqdm(models):
    tqdm.write(f'Training {name}...')
    kfold = KFold(n_splits=10, random_state=42)
    cv_f1_scores = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='f1')
    results.append(cv_f1_scores)
    names.append(name)
    tqdm.write(f'{name}: {cv_f1_scores.mean()} ({cv_f1_scores.std()})')
