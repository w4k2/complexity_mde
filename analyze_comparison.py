import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from scipy.ndimage import gaussian_filter1d


# DATASETS x FOLDS x TRANSFER (imagenet | best BAC | best transrate) x EPOCH
scores = np.load("results/transfer/comparison.npy")
scores = scores[:14]
# RANSFER (imagenet | best BAC | best transrate) x EPOCH
mean_scores = np.mean(scores, axis=1)

transfer_names = ["imagenet", "best BAC", "best transrate"]

for data_id in range(mean_scores.shape[0]):
    print(mean_scores[data_id, :, -1])
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    for i, scores in enumerate(mean_scores[data_id]):
        ax.plot(gaussian_filter1d(scores, 3), label=transfer_names[i])

    plt.legend()
    plt.tight_layout()
    plt.savefig("foo.png")
    sleep(2)
    plt.close()
