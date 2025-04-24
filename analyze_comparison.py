import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from scipy.ndimage import gaussian_filter1d


# DATASETS x FOLDS x TRANSFER (imagenet | best BAC | best transrate) x EPOCH
scores = np.load("results/transfer/comparison_banknote.npy")

# scores = scores[:2]
# RANSFER (imagenet | best BAC | best transrate) x EPOCH
mean_scores = np.mean(scores, axis=1)

transfer_names = ["imagenet", "best BAC", "best transrate"]

for data_id in range(mean_scores.shape[0]):
    data_id += 2
    print(mean_scores[data_id, :, -1])
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    for i, scores in enumerate(mean_scores[data_id]):
        ax.plot(gaussian_filter1d(scores, 3), label=transfer_names[i])
    
    ax.set_xticks(np.arange(0, 20, 1), [str(i+1) for i in np.arange(0, 20, 1)])
    ax.set_xlim(-.2, 19.2)
    ax.grid(ls=":", c=(.7, .7, .7))

    plt.legend()
    plt.tight_layout()
    plt.savefig("foo_banknote2.png")
    sleep(2)
    plt.close()
    exit()
