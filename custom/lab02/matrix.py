import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import pandas as pd
import matplotlib.pyplot as plt


#%matplotlib inline


class Matrix(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.filedict = []
        self.filelength = 0

    def ParseFile(self):
        self.filedict = np.loadtxt(open(self.filepath, "rb"), delimiter=",")
        self.filelength = len(self.filedict)

    def ParseImageFile(self, imagepath, extension=".jpg"):
        dataset = pd.read_csv(self.filepath)
        return dataset


    def getPrimitive(self, *args):

        if len(args) == 0:
            return self.filedict

        arg = [item for item in args]

        prims = []
        for line in self.filedict:
            #tmp = np.array(line.split(','))
            #prim = tmp[arg]
            prim = line[arg]
            prims.append(list(prim))

        return prims

    def FloatArray(self, array):
        tmp = np.array(array)
        return tmp.astype(float)

    # ratio: le ratio en % du test set a evaluer (INT)
    def CreateTestSet(self, dataset, ratio):
        ratio = int(ratio)
        if ratio >= 100:
            return

        # nombre de donner de test a extraire
        k = int(self.filelength * (ratio / 100))
        
        random.shuffle(dataset)
        return np.array(dataset[:k]), np.array(dataset[k:])

    def PrintMetricsScore(self, y_true, y_pred, modele):
        ascore = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        print("{0} scores: Accuracy: {1} ; F1: {2}".format(modele, ascore, f1))
        return ascore, f1

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

