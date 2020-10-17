import matplotlib.pyplot as plt
import numpy as np

def ReadData(fileName):
    with open(fileName, 'r') as f:
        results = np.empty(20, dtype=np.int64)
        for i in range(0, 20):
            results[i] = int(f.readline())
        avgMistakes = np.average(results)
        stdDev = np.std(results)
        return (avgMistakes, stdDev)

# Experiment 1 filenames
fileNames1 = ['S500_A10_F0', 'S500_A20_F0', 'S500_A30_F0', 'S500_A40_F0', 'S500_A50_F0', 'S500_A60_F0', 'S500_A70_F0', 'S500_A80_F0', 'S500_A90_F0']

X = [10, 20, 30, 40, 50, 60, 70, 80, 90]
Y = np.empty(len(fileNames1), dtype=np.float32)
StdDev = np.empty(len(fileNames1), dtype=np.float32)

for i in range(0, len(fileNames1)):
    (Y[i], StdDev[i]) = ReadData(fileNames1[i])

print(Y)
print(StdDev)
fig = plt.figure()
ax = plt.axes()

plt.errorbar(X, Y, StdDev, linestyle=':', capsize=3, marker='o', elinewidth=1);
plt.xlabel("Triangle Angle (A)")
plt.ylabel("Number of misclassified points")

plt.savefig("Experiment_3_Figure")
plt.show()
