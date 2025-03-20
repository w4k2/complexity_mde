from sklearn.datasets import make_classification
import numpy as np
from mde import STML, STML_RGB
import matplotlib.pyplot as plt



X, y = make_classification(n_samples=10, n_features=8, n_informative=8, n_redundant=0, random_state=1410)
print(X[0])

stml = STML()
X_stml = stml.fit_transform(X)

stml_rgb = STML_RGB()
X_stml_rgb = stml_rgb.fit_transform(X, y)

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(X_stml[0])


X_stml_rgb = stml_rgb.transform(X)
ax[1].imshow(X_stml_rgb[0])

plt.tight_layout()
plt.savefig("foo.png")
