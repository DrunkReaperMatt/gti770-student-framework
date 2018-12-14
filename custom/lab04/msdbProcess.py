import sys, getopt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from custom.lab04.traitement import Traitement as tr


def main(argv):
    X = []
    trackid = []
    clf = VotingClassifier()
    lbl = LabelEncoder()
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "mfile=", "lfile="])
    except getopt.GetoptError:
        print('msdbProcess.py -i <inputfile>.csv -m <models>.joblib -l <labelencode>.joblib')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('msdbProcess.py -i <inputfile>.csv -m <models>.joblib -l <labelencode>.joblib')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            dataset = np.genfromtxt(arg, delimiter=',')
            trackid, X = preproc(dataset)
        elif opt in ("-m", "--mfile"):
            clf = tr.ImportModel(arg)
        elif opt in ("-l", "--lfile"):
            lbl = tr.ImportModel(arg)
    Y = predict(clf, X)
    Y = lbl.inverse_transform(Y)
    output(trackid, Y)
    sys.exit(2)


def preproc(dataset):
    trackid = []
    X = []

    for data in dataset:
        trackid.append(data[0])
        X.append(data[1:-1])

    trackid = np.array(trackid)
    X = np.array(X)
    X = tr.MinMax(X)
    return trackid, X


def predict(clf, X):
    return clf.predict(X)


def output(trackid, y):
    with open("output.txt", 'w') as f:
        f.write("id, genre\n")
        for i in range(len(trackid)):
            f.write("{}, {}\n".format(trackid[i], y[i]))


if __name__ == "__main__":
    main(sys.argv[1:])

