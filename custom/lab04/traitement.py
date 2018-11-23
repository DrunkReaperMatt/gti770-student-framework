import numpy as np
import pandas as pd
from custom.lab04.train import Train
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, LabelEncoder


class Traitement(object):

    def Dataframe(self, path, columns):
        return pd.read_csv(path, names=columns, delimiter=',')

    def ParseValidSet(self, dataset):
        X = [], Y = []
        for line in dataset:
            X.append(line[:-1])
            Y.append(line[-1])

        return X, Y

    def MinMax(self, X):
        mms = MinMaxScaler()
        return mms.fit_transform(X)

    def Kbin(self, X):
        kbs = KBinsDiscretizer()
        return kbs.fit_transform(X)

    def Labels(self, Y):
        lbl = LabelEncoder()
        return lbl.fit(Y)
