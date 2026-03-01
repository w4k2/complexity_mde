import numpy as np


def hscore(features: np.ndarray, labels: np.ndarray):
    f = features
    y = labels

    covf = np.cov(f, rowvar=False)
    C = int(y.max() + 1)
    g = np.zeros_like(f)

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = np.cov(g, rowvar=False)
    score = np.trace(np.dot(np.linalg.pinv(covf, rcond=1e-15), covg))

    return score

import numpy as np

def hscore_fixed(features: np.ndarray, labels: np.ndarray, ridge: float = 1e-4):
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)

    good = np.isfinite(f).all(axis=1)
    f = f[good]
    y = y[good]

    classes = np.unique(y)

    covf = np.cov(f, rowvar=False, bias=True)
    D = covf.shape[0]
    covf = covf + ridge * np.eye(D)

    g = np.zeros_like(f)
    for c in classes:
        idx = (y == c)
        g[idx] = f[idx].mean(axis=0)

    covg = np.cov(g, rowvar=False, bias=True)

    score = float(np.trace(np.linalg.pinv(covf) @ covg))
    return score
