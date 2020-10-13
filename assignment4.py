import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from math import cos, sin, radians, sqrt

# Triangle coordinates
T0 = (3, 3)
T1 = (7, 3)
T2 = (7, 7)

# Draws the scatterplot of points X with colors Y (blue if True, red if False)
def DrawPlot(X, Y, T0, T1, T2):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    # Draw the points in the plot
    for i in range(0, len(Y)):
        if (Y[i]):
            plt.scatter(X[i][0], X[i][1], s=10, color='blue')
        else:
            plt.scatter(X[i][0], X[i][1], s=10, color='red')

    # Draw the triangle lines
    lines = [[T0, T1], [T1, T2], [T0, T2]]
    lc = mc.LineCollection(lines, colors='black', linewidths=1)
    ax.add_collection(lc)

    # Draw the plot
    ax.set_aspect('equal', adjustable='box')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()

# Compute whether point is in triangle using barycentric coordinates
def PointInTriangle(P, T0, T1, T2):
    d = ((T1[1]-T2[1]) * (T0[0]-T2[0]) + (T2[0]-T1[0]) * (T0[1]-T2[1]))
    a = ((T1[1]-T2[1]) * (P[0]-T2[0]) + (T2[0]-T1[0]) * (P[1] - T2[1])) / d
    b = ((T2[1]-T0[1]) * (P[0]-T2[0]) + (T0[0]-T2[0]) * (P[1] - T2[1])) / d
    c = 1 - a - b

    return True if (0 <= a <= 1 and  0 <= b <= 1 and 0 <= c <= 1) else False

# Compute distance between two points
def ComputeDistance(a, b):
    return np.linalg.norm(np.subtract(a, b))

# Classifies points in X using KNN and returns their labels in Y
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

# Count how many labels in Y mismatch with the ground truth
def CountMisclassified(X, Y):
    misclassified = 0
    for i in range(0, len(Y)):
        x = X[i]
        if (Y[i] != PointInTriangle(x, T0, T1, T2)):
            misclassified += 1
    return misclassified

# Run one experiment with given parameters
def RunExperiment(n, f, k, T0, T1, T2):
    S = np.random.uniform(0, 10, (n, 2))

    L = []
    for i in range(0, n):
        P = S[i]
        L.append(PointInTriangle(P, T0, T1, T2))

    X = np.random.uniform(0, 10, (10000, 2))
    Y = KNN(X, S, L, k)
    misclassified = CountMisclassified(X, Y)
    print(misclassified)
    return misclassified

# Write results to file
def WriteResultsToFile(n, a, f, results):
    file_name = "S"+str(n)+"_A"+str(a)+"_F"+str(f)
    with open(file_name, 'w') as f:
        for i in range(0, len(results)):
            f.write(str(results[i])+'\n')

def FindTriangleCoords(T0, angle):
    a = radians(angle)
    T = 8
    s = sqrt(T / (0.5 * sin(a)))
    return ((T0[0] + s, T0[1]), (T0[0] + s * cos(a), T0[1] + s * sin(a)))

def RunExperiment1():
    for n in range(100, 801, 100):
        print("N: " + str(n))
        results = []
        for i in range(0, 20):
            results.append(RunExperiment(n, 0, 5, T0, T1, T2))

        WriteResultsToFile(n, 5, 0, results)

def RunExperiment3():
    T0 = (0, 0)
    for a in range(10, 91, 10):
        print("Angle: " + str(a))
        (T1, T2) = FindTriangleCoords(T0, a)
        results = []
        for i in range(0, 20):
            results.append(RunExperiment(500, 0, 5, T0, T1, T2))

        WriteResultsToFile(500, a, 0, results)

RunExperiment3()
