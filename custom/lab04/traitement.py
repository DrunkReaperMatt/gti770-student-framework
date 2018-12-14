import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, LabelEncoder
from sklearn.decomposition import PCA
from joblib import dump, load


class Traitement(object):
    def __init__(self):
        self.lbl = LabelEncoder()

    def Dataframe(self, path, primitives):
        index = ['SAMPLEID', 'TRACKID']
        for i in range(primitives):
            index.append('component_{}'.format(i))
        index.append('class')

        return pd.read_csv(path, names=index, delimiter=',')

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

    def ExportModel(self, clf, filename):
        dump(clf, "models/{}.joblib".format(filename))

    def ImportModel(self, filename):
        return load("{}.joblib".format(filename))


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