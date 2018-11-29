from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard

class Bayes(object):

    def __init__(self):
        self.gnb = GaussianNB()
        self.mnb = MultinomialNB()

    # Gaussian Naive Bayes
    def GaussNB(self, X, Y, trainX):
        self.gnb.fit(X, Y)
        return self.gnb.predict(trainX)


    # Multinomial Naive Bayes
    def Multinomial(self, X, Y, trainX):
        self.mnb.fit(X, Y)
        return self.mnb.predict(trainX)

    ##MinMax Scaler
    def MinMaxNB(self, X, Y, trainX):
        mms = MinMaxScaler()
        X_minmax = mms.fit_transform(X)
        train_minmax = mms.fit_transform(trainX)
        self.mnb.fit(X_minmax, Y)
        return self.mnb.predict(train_minmax)

    ##Dicrestisation non-superviser
    def DiscretNonSupp(self, X, Y, trainX):
        #kbd = KBinsDiscretizer()
        kbd = MinMaxScaler()
        X_KBD = kbd.fit_transform(X)
        train_KBD = kbd.fit_transform(trainX)
        self.mnb.fit(X_KBD, Y)
        return self.mnb.predict(train_KBD)


class KNN(object):

    def Uniform(self, X, Y, trainX, k=1):
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        knn.fit(X, Y)
        return knn.predict(trainX)

    def Distance(self, X, Y, trainX, k=1):
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(X, Y)
        return knn.predict(trainX)


class DecisionTree(object):

    def DepthNone(self, X, Y, trainX):
        dtc = DecisionTreeClassifier()
        dtc.fit(X, Y)
        return dtc.predict(trainX)

    def DepthK(self, X, Y, trainX, k):
        dtc = DecisionTreeClassifier(max_depth=k)
        dtc.fit(X, Y)
        return dtc.predict(trainX)


class NeuralNet(object):

    def Model(self, perceptron=100, layers=2, learn_rate=0.0005):
        model = keras.Sequential()

        for i in range(layers):
            model.add(keras.layers.Dense(perceptron))

        model.add(keras.layers.Dense(2))
        model.compile(tf.train.AdamOptimizer(learn_rate),
                      loss='mse', metrics=['accuracy'])
        return model

    def getCallback(self, runName='default'):
        return TensorBoard(log_dir='./logs/{}'.format(runName), write_graph=False, write_images=True)


class MachineVecteur(object):

    def Linear(self, c=1e-03):
        svc = SVC(kernel='linear', C=c, class_weight='balanced')
        return svc

    def RBF(self, c=1e-3, gamma=1e-3):
        svc = SVC(kernel='rbf', gamma=gamma, C=c)
        return svc


class CrossValidation(object):

    def AccuracyScore(self, clf, X, Y, cv=10):
        score = cross_val_score(clf, X, Y, cv=cv)
        return score.mean()

    def F1Score(self, clf, X, Y, cv=10):
        score = cross_val_score(clf, X, Y, cv=cv, scoring='f1_macro')
        return score.mean()


class Class(object):

    def Gauss(self):
        return GaussianNB()

    def Multi(self):
        return MultinomialNB()

    def KnnUni(self, k):
        return KNeighborsClassifier(n_neighbors=k, weights='uniform')

    def KnnDist(self, k):
        return KNeighborsClassifier(n_neighbors=k, weights='distance')

    def Tree0(self):
        return DecisionTreeClassifier()

    def TreeK(self, k):
        return DecisionTreeClassifier(max_depth=k)
