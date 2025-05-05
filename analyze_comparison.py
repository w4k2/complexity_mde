import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from scipy.ndimage import gaussian_filter1d


dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']

# DATASETS x FOLDS x TRANSFER (imagenet | best BAC | best transrate) x EPOCH
# scores = np.load("results/transfer/comparison_imgnet_extraction.npy")
scores = np.load("results/transfer/comparison_imgnet_finetuning.npy")

# DATASETS x TRANSFER (imagenet | best BAC | best transrate) x EPOCH
mean_scores = np.mean(scores, axis=1)

transfer_names = ["imagenet", "best BAC", "best transrate"]

# for each dataset
# for data_id in range(mean_scores.shape[0]):
#     fig, ax = plt.subplots(1, 1, figsize=(15, 10))

#     for i, scores in enumerate(mean_scores[data_id]):
#         ax.plot(gaussian_filter1d(scores, 3), label=transfer_names[i])
    
#     ax.set_xticks(np.arange(0, 50, 1), [str(i+1) for i in np.arange(0, 50, 1)])
#     ax.set_xlim(-.2, 49.2)
#     ax.grid(ls=":", c=(.7, .7, .7))
#     ax.set_title(dataset_names[data_id])

#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("foo.png")
#     sleep(2)
#     plt.close()

# mean plot for all datasets
# TRANSFER (imagenet | best BAC | best transrate) x EPOCH
data_mean_scores = np.mean(mean_scores, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

for i, scores in enumerate(data_mean_scores):
    ax.plot(gaussian_filter1d(scores, 3), label=transfer_names[i])

ax.set_xticks(np.arange(0, 50, 1), [str(i+1) for i in np.arange(0, 50, 1)])
ax.set_xlim(-.2, 49.2)
ax.grid(ls=":", c=(.7, .7, .7))

plt.legend()
plt.tight_layout()
plt.savefig("bar.png")
plt.close()