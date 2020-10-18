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
fileNames1 = ['S100_A0_F0', 'S200_A0_F0', 'S300_A0_F0', 'S400_A0_F0', 'S500_A0_F0', 'S600_A0_F0', 'S700_A0_F0', 'S800_A0_F0']

X = [100, 200, 300, 400, 500, 600, 700, 800]
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
plt.xlabel("Density of S (n)")
plt.ylabel("Number of misclassified points")

plt.grid(True, **kwargs)
plt.savefig("Experiment_1_Figure")
plt.show()
