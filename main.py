#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn import preprocessing
import random


###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #

featuresToUse = "CaffeNet4096"  # surf, CaffeNet4096, GoogleNet1024
numberIteration = 10
adaptationAlgoUsed = ["NA", "SA"]
# see function adaptData for available algorithms

#                                                                             #
#               End of part of code about arguments to modify                 #
###############################################################################


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def split(Y, nPerClass):
    idx1 = []
    idx2 = []
    for c in range(1, max(Y)+1):
        idx = indices(Y, lambda x: x == c)
        random.shuffle(idx)
        idx1 = idx1 + idx[0:min(nPerClass, len(idx))]
        idx2 = idx2 + idx[min(nPerClass, len(idx)):-1]
    return idx1, idx2


def adaptData(algo, sourceData, sourceLabels, targetData, targetLabels):
    if algo == "NA":  # No Adaptation
        sourceAdapted = sourceData
        targetAdapted = targetData
    if algo == "SA":
        # Subspace Alignment, described in:
        # Unsupervised Visual Domain Adaptation Using Subspace Alignment, 2013,
        # Fernando et al.
        from sklearn.decomposition import PCA
        d = 80  # subspace dimennsion
        pcaS = PCA(d).fit(sourceData)
        pcaT = PCA(d).fit(targetData)
        XS = np.transpose(pcaS.components_)[:, :d]  # source subspace matrix
        XT = np.transpose(pcaT.components_)[:, :d]  # target subspace matrix
        Xa = XS.dot(np.transpose(XS)).dot(XT)  # align source subspace
        sourceAdapted = sourceData.dot(Xa) # poject source in aligned subspace
        targetAdapted = targetData.dot(XT) # project target in target subspace

    return sourceAdapted, targetAdapted


def getAccuracy(trainData, trainLabels, testData, testLabels):
    # ------------ Accuracy evaluation by performing a 1NearestNeighbor
    dist = cdist(trainData, testData, metric='sqeuclidean')
    minIDX = np.argmin(dist, axis=0)
    prediction = trainLabels[minIDX]
    accuracy = 100 * float(sum(prediction == testLabels)) / len(testData)
    return accuracy

# ---------------------------- DATA Loading Part ------------------------------
domainNames = ['amazon', 'caltech10', 'dslr', 'webcam']
tests = []
data = {}

for sourceDomain in domainNames:
    possible_data = loadmat(os.path.join(".", "features", featuresToUse,
                                         sourceDomain + '.mat'))
    if featuresToUse == "surf":
        # Normalize the surf histograms
        feat = (possible_data['fts'].astype(float) /
                np.tile(np.sum(possible_data['fts'], 1),
                        (np.shape(possible_data['fts'])[1], 1)).T)
    else:
        feat = possible_data['fts'].astype(float)

    # Z-score
    feat = preprocessing.scale(feat)

    labels = possible_data['labels'].ravel()
    data[sourceDomain] = [feat, labels]
    for targetDomain in domainNames:
        if sourceDomain != targetDomain:
            perClassSource = 20
            if sourceDomain == 'dslr':
                perClassSource = 8
            tests.append([sourceDomain, targetDomain, perClassSource])

meansAcc = {}
stdsAcc = {}
print("Feature used: ", featuresToUse)
print("Number of iterations: ", numberIteration)
print("Adaptation algorithms used: ", end="")
for name in adaptationAlgoUsed:
    meansAcc[name] = []
    stdsAcc[name] = []
    print(" ", name, end="")
print("")

# -------------------- Main testing loop --------------------------------------
for test in tests:
    startTime = time.time()
    Sname = test[0]
    Tname = test[1]
    perClassSource = test[2]
    print(Sname.upper()[:1] + '->' + Tname.upper()[:1], end=" ")

    # --------------------II. prepare data-------------------------------------
    Sx = data[Sname][0]
    Sy = data[Sname][1]
    Tx = data[Tname][0]
    Ty = data[Tname][1]

    # --------------------III. run experiments---------------------------------
    results = {}
    for name in adaptationAlgoUsed:
        results[name] = []
    for iteration in range(numberIteration):
        id1, id2 = split(Sy, perClassSource)
        subSx = Sx[id1, :]
        subSy = Sy[id1]

        for name in adaptationAlgoUsed:
            # Apply domain adaptation algorithm
            subSa, Ta = adaptData(name, subSx, subSy, Tx, Ty)
            # Compute the accuracy classification
            results[name].append(getAccuracy(subSa, subSy, Ta, Ty))
        print(".", end="")

    currentTime = time.time()
    print(" {:6.2f}".format(currentTime - startTime) + "s")

    for name in adaptationAlgoUsed:
        meanAcc = np.mean(results[name])
        stdAcc = np.std(results[name])
        meansAcc[name].append(meanAcc)
        stdsAcc[name].append(stdAcc)
        print("     {:4.1f}".format(meanAcc), " {:3.1f}".format(stdAcc), name)

print("")
print("Mean results:")
for name in adaptationAlgoUsed:
    meanMean = np.mean(meansAcc[name])
    meanStd = np.mean(stdsAcc[name])
    print("     {:4.1f}".format(meanMean), " {:3.1f}".format(meanStd), name)
