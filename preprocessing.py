import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from numpy import hstack
from sklearn.decomposition import TruncatedSVD

class FeatureHashing():
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.MaxHash = 10000
        self.conam_enc = MinMaxScaler()
        self.label_encs = None
        self.onehot_enc = OneHotEncoder(categories='auto')
        self.svd = TruncatedSVD(n_components=5000, algorithm='arpack')
        self._fit()

    def _fit(self):
        train_set = pd.read_csv(self.train_set)
        test_set = pd.read_csv(self.test_set)
        
        Y = train_set['fraud_ind']
        X = train_set.drop(columns=['fraud_ind']).append(test_set, ignore_index=True)
        self.conam_enc.fit(X['conam'].values.reshape(-1, 1)).astype('float16')
        X = X.drop(columns=['txkey', 'locdt', 'conam'])
        X = X.astype(str)

        keys = list(X.keys())
        self.label_encs = dict(zip(keys, [LabelEncoder() for _ in keys]))

        for key in keys:
            X = self.label_encs[key].fit_transform(X[key].values)

        X = self._hash_func(X)
        X = self.onehot_enc.fit_transform(X)
        X = self.svd.fit_transform(X)

        return self

    def training_set(self):
        train_set = pd.read_csv(self.train_set)
        
        Y = train_set['fraud_ind']
        X = train_set.drop(columns=['fraud_ind'])
        conam = self.conam_enc.transform(X['conam'].values.reshape(-1, 1)).astype('float16')
        X = X.drop(columns=['txkey', 'locdt', 'conam'])
        X = X.astype(str)

        keys = list(X.keys())
        self.label_encs = dict(zip(keys, [LabelEncoder() for _ in keys]))

        for key in keys:
            X[key] = self.label_encs[key].transform(X[key].values)

        X = self._hash_func(X)
        X = self.onehot_enc.transform(X)
        X = self.svd.transform(X)
        X = hstack((X, conam))

        return X, Y

    def testing_set(self):
        X = pd.read_csv(self.test_set)
        conam = self.conam_enc.transform(X['conam'].values.reshape(-1, 1)).astype('float16')
        txkey = X[['txkey']]
        X = X.drop(columns=['txkey', 'locdt', 'conam'])
        X = X.astype(str)
        keys = list(X.keys())

        for key in keys:
            X[key] = self.label_encs[key].transform(X[key].values)

        X = self._hash_func(X)
        X = self.onehot_enc.transform(X)
        X = self.svd.transform(X)
        X = hstack((X, conam))

        return X, txkey

    def _hash_func(self, x):
        return x % self.MaxHash
