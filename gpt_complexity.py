import numpy as np
import problexity as px


data = np.genfromtxt('gpt/gpt_dataset.csv', delimiter=',', skip_header=1)
X, y = data[:, :-1].astype(float), data[:, -1].astype(int)
print(X)
print(y)