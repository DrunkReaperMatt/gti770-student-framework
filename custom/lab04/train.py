import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import VotingClassifier


class Train(object):

    ## Decision Tree
    def DepthK(self, k):
        return DecisionTreeClassifier(max_depth=k)

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

    ## Classe de vote
    def Vote(self, X, Y, classifiers):
        vclf = VotingClassifier(classifiers, n_jobs=-1)
        return vclf.fit(X, Y)

    def AccuracyScore(self, clf, X, Y, cv=10):
        score = cross_val_score(clf, X, Y, cv=cv)
        return score.mean()

    def F1Score(self, clf, X, Y, cv=10):
        score = cross_val_score(clf, X, Y, cv=cv, scoring='f1_macro')
        return score.mean()

    ## Too powerfull, do not use
    def ParameterOptimizer(self, X, Y, cv=10):
        svmhp = {'C': np.logspace(-3, 6, 10), 'gamma': np.logspace(-8, 1, 10)}
        svc = SVC(kernel="rbf")
        gcv = GridSearchCV(svc, svmhp, cv=cv)
        gcv.fit(X, Y)

        return gcv

