from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import os
from KNN_KDTree import KDTree_like_sklearn

def get_score(y_predict, y_test):
    l = len(y_test)
    ans_eql = 0
    for i in range(l):
        if(y_predict[i] == y_test[i]):
            ans_eql += 1
    return ans_eql * 1.0 / l

def evaluate_classifier(clf, testX, testY):
    y_pre = clf.predict(testX)
    return get_score(y_pre, testY)

def evaluate_classifier_parameters(clf, data, target, split_ratio):
    trainX, testX, trainY, testY = train_test_split(data, target, train_size=split_ratio, random_state=0)
    clf.fit(trainX, trainY)
    y_pre = clf.predict(testX)
    return get_score(y_pre, testY)

def load_train_data():
    trainX = list()
    trainY = list()
    dir_path = './digits/trainingDigits'
    for filename in os.listdir(dir_path):
        # print(filename)
        trainY.append(int(filename[0]))
        t = np.loadtxt(os.path.join(dir_path, filename), str)
        tt = list(np.zeros(32 * 32))
        for i in range(32):
            for j in range(32):
                if t[i][j] == '1':
                    tt[i * 32 + j] = 1
                else:
                    tt[i * 32 + j] = 0
        trainX.append(tt)
    return trainX, trainY
    # print(trainX[0:2])


def load_test_data():
    testX = list()
    testY = list()
    dir_path = './digits/testDigits'
    for filename in os.listdir(dir_path):
        # print(filename)
        testY.append(int(filename[0]))
        t = np.loadtxt(os.path.join(dir_path, filename), str)
        tt = np.zeros(32 * 32)
        for i in range(32):
            for j in range(32):
                if t[i][j] == '1':
                    tt[i * 32 + j] = 1
        testX.append(tt)
    return testX, testY

load_train_data()

trainX, trainY = load_train_data()
testX, testY = load_test_data()

# pca = PCA(n_components=32)
# pca.fit(trainX)
# trainX = pca.transform(trainX)

# clf = KNeighborsClassifier(n_neighbors=5)
# score_t = evaluate_classifier_parameters(clf, trainX, trainY, 0.99)
# print(score_t)
# score = evaluate_classifier(clf, pca.transform(testX), testY)
# print(score)

clf2 = KDTree_like_sklearn(dist_kind='simple', k=5)
score_t = evaluate_classifier_parameters(clf2, trainX, trainY, 0.99)
print(score_t)
