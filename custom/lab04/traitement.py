import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, LabelEncoder
from sklearn.decomposition import PCA


class Traitement(object):
    def __init__(self):
        self.lbl = LabelEncoder()

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

    def Encoder(self, Y):
        return self.lbl.fit_transform(Y)

    def Decoder(self, Y):
        return self.lbl.inverse_transform(Y)

    def Shrinking(self, components):
        pca = PCA(n_components=components, svd_solver='auto')
        return pca

    def PrintPlot(self, accscore, f1score, labels, title="Plot"):
        df = pd.DataFrame(dict(x=accscore, y=f1score, label=labels))
        groups = df.groupby('label')

        fig, axe = plt.subplots()

        for name, group in groups:
            axe.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
        axe.legend(numpoints=1, loc='upper left')
        plt.xlabel("Accuracy Score")
        plt.ylabel("F1 Score")
        plt.title(title)
        plt.show()