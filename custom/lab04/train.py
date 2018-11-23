import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
import scipy


class Train(object):

    ## KNN
    def Uniform(self, X, Y, trainX, k=1):
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto')
        knn.fit(X, Y)
        return knn.predict(trainX)

    def Distance(self, X, Y, trainX, k=1):
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto')
        knn.fit(X, Y)
        return knn.predict(trainX)

    ## Bayes
    # Gaussian Naive Bayes
    def GaussNB(self, X, Y, trainX):
        gnb = GaussianNB()
        gnb.fit(X, Y)
        return gnb.predict(trainX)

    # Multinomial Naive Bayes
    def Multinomial(self, X, Y, trainX):
        mnb = MultinomialNB()
        mnb.fit(X, Y)
        return mnb.predict(trainX)

    # Bayes Bernoulli
    def Bernoulli(self, X, Y, trainX):
        ber = BernoulliNB()
        ber.fit(X, Y)
        return ber.predict(trainX)

    ## SVM
    def Linear(self, X, Y, trainX, c=1e-03, weight='balanced'):
        svc = SVC(kernel='linear', C=c, class_weight=weight)
        svc.fit(X, Y)
        return svc.predict(trainX)

    def RBF(self, X, Y, trainX, c=1e-3, gamma=1e-3, weight='balanced'):
        svc = SVC(kernel='rbf', gamma=gamma, C=c, class_weight=weight)
        svc.fit(X, Y)
        return svc.predict(trainX)

    def OtherLinear(self, X, Y, trainX, c=1e-3, weight='balanced'):
        svc = LinearSVC(C=c, class_weight=weight)
        svc.fit(X, Y)
        return svc.predict(trainX)

    def ParameterOptimizer(self, X, Y, cv=10):
        svmhp = {'C': scipy.expon(scale=100), 'gamma': scipy.expon(scale=.1), 'kernel': ['rbf', 'linear'], 'class_weight': ['balanced', None]}
        svc = SVC()
        gcv = GridSearchCV(svc, svmhp, cv=cv)
        gcv.fit(X, Y)

        return gcv.cv_results_
