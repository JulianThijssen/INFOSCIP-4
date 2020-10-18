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

# Experiment 2 filenames
fileNames1 = ['S500_A0.0_F0', 'S500_A0.05_F0', 'S500_A0.1_F0', 'S500_A0.15_F0', 'S500_A0.2_F0', 'S500_A0.25_F0', 'S500_A0.3_F0']

X = np.arange(0.0, 0.31, 0.05)
Y = np.empty(len(fileNames1), dtype=np.float32)
StdDev = np.empty(len(fileNames1), dtype=np.float32)

for i in range(0, len(fileNames1)):
    (Y[i], StdDev[i]) = ReadData(fileNames1[i])

print(Y)
print(StdDev)
fig = plt.figure()
ax = plt.axes()

kwargs = {'linestyle' : '--', 'alpha' : 0.5}
plt.errorbar(X, Y, StdDev, linestyle=':', capsize=3, marker='o', elinewidth=1);
plt.xlabel("Outlier fraction (f)")
plt.ylabel("Number of misclassified points")

plt.grid(True, **kwargs)
plt.savefig("Experiment_2_Figure")
plt.show()
