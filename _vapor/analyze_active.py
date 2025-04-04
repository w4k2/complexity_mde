import numpy as np
import matplotlib.pyplot as plt


# DATASETS x FOLDS x METHOD x QUERIES
scores = np.load("results/active/preliminary_full.npy")[:, :, :, :100]
# METHOD x QUERIES
mean_scores = np.mean(scores[14], axis=0)

method_names = ["TAB", "STML", "CROSS"]

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
for method_id in range(3):
    ax.plot(mean_scores[method_id], label=method_names[method_id])

plt.legend()
plt.tight_layout()
plt.savefig("foo.png")