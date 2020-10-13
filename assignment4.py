import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from collections import Counter

n = 500
f = 0

T0 = (3, 3)
T1 = (7, 3)
T2 = (7, 7)

def PointInTriangle(P, T0, T1, T2):
    d = ((T1[1]-T2[1]) * (T0[0]-T2[0]) + (T2[0]-T1[0]) * (T0[1]-T2[1]))
    a = ((T1[1]-T2[1]) * (P[0]-T2[0]) + (T2[0]-T1[0]) * (P[1] - T2[1])) / d
    b = ((T2[1]-T0[1]) * (P[0]-T2[0]) + (T0[0]-T2[0]) * (P[1] - T2[1])) / d
    c = 1 - a - b

    return True if (0 <= a <= 1 and  0 <= b <= 1 and 0 <= c <= 1) else False

def DrawPlot(X, Y, T0, T1, T2):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    for i in range(0, len(Y)):
        if (Y[i]):
            plt.scatter(X[i][0], X[i][1], s=10, color='blue')
        else:
            plt.scatter(X[i][0], X[i][1], s=10, color='red')

    lines = [[T0, T1], [T1, T2], [T0, T2]]
    lc = mc.LineCollection(lines, colors='black', linewidths=1)
    ax.add_collection(lc)
    ax.set_aspect('equal', adjustable='box')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()

def ComputeDistance(a, b):
    return np.linalg.norm(np.subtract(a, b))

def KNN(X, S, L, k):
    Y = np.empty(len(X), dtype=bool)
    for i in range(0, len(X)):
        x = X[i]
        dists = np.empty(len(S))
        for j in range(0, len(S)):
            s = S[j]
            dists[j] = ComputeDistance(x, s)
        indices = np.argsort(dists)
        neighbours = [L[index] for index in indices[0:k]]
        Y[i] = np.bincount(neighbours).argmax()
    return Y

def CountMisclassified(X, Y):
    misclassified = 0
    for i in range(0, len(Y)):
        x = X[i]
        if (Y[i] != PointInTriangle(x, T0, T1, T2)):
            misclassified += 1
    return misclassified

def RunExperiment(n, f, T0, T1, T2, k):
    S = np.random.uniform(0, 10, (n, 2))

    L = []
    for i in range(0, n):
        P = S[i]
        L.append(PointInTriangle(P, T0, T1, T2))

    X = np.random.uniform(0, 10, (1000, 2))
    
    Y = KNN(X, S, L, k)
    misclassified = CountMisclassified(X, Y)
    print(misclassified)
    #DrawPlot(X, Y, T0, T1, T2)

RunExperiment(n, f, T0, T1, T2, 7)
