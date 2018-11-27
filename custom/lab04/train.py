import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier


class Train(object):

    ## KNN
    def Uniform(self, k=1):
        return KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto')

    def Distance(self, k=1):
        return KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto')

    ## Bayes
    # Gaussian Naive Bayes
    def GaussNB(self):
        return GaussianNB()

    # Multinomial Naive Bayes
    def Multinomial(self):
        return MultinomialNB()

    # Bayes Bernoulli
    def Bernoulli(self):
        return BernoulliNB()

    ## SVM
    def Linear(self, c=1e-03, weight='balanced'):
        return SVC(kernel='linear', C=c, class_weight=weight)

    def RBF(self, c=1e-3, gamma=1e-3, weight='balanced'):
        return SVC(kernel='rbf', gamma=gamma, C=c, class_weight=weight)

    def Vote(self, X, Y, trainX, classifiers):
        vclf = VotingClassifier(classifiers, n_jobs=-1)
        vclf.fit(X, Y)
        return vclf.predict(trainX)


    ## Too powerfull, do not use
    def ParameterOptimizer(self, X, Y, cv=10):
        svmhp = {'C': [1, 10], 'gamma': [.1, 2], 'kernel': ['rbf', 'linear'], 'class_weight': ['balanced', None]}
        svc = SVC()
        gcv = GridSearchCV(svc, svmhp, cv=cv)
        gcv.fit(X, Y)

        return gcv.cv_results_

