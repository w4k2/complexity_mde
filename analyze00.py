import numpy as np
import matplotlib.pyplot as plt
from utils import Data
from scipy.stats import rankdata
# Stat
from utils import cv52cft
from scipy.stats import rankdata, wilcoxon
from tabulate import tabulate
import matplotlib
matplotlib.rcParams.update({'font.size': 22, "font.family" : "monospace"})


data = Data(selection=("all", ["balanced", "binary"]), path="datasets/")
datasets = data.load()
datasets = list(datasets.keys())
encodings = ["STML", "STML_RGB", "XGB"]

scores_stml = np.load("results/ex00_preliminary_stml_transfer_224_scores.npy")
scores_xgb = np.load("results/xgb_scores.npy")

# ENCODING x DATASETS x FOLDS x EPOCHS
gathered = np.stack((scores_stml, scores_stml))
ttest_data = gathered[:, :, :, -1]
ttest_data = np.concatenate((ttest_data, scores_xgb.reshape(1, 22, 10)), axis=0)

# ENCODING x DATASETS
gathered = np.mean(gathered, axis=2)[:, :, -1]
gathered = np.concatenate((gathered, np.mean(scores_xgb, axis=1).reshape(1, 22)), axis=0)

# DATASETS x ENCODING
gathered = gathered.T
wilcoxon_data = gathered

# print("Mean BAC:  ", np.mean(gathered, axis=0))
# print("Mean rank: ", np.mean(rankdata(gathered, axis=1), axis=0))

x = np.arange(len(datasets))  # the label locations
width = 0.16  # the width of the bars
multiplier = -1

fig, ax = plt.subplots(1, 1, figsize=(18, 10))
colors = ["lightsteelblue", "moccasin", "darkseagreen", "tomato", "lightgrey"]
# colors = ["#7CB6F0", "#EEA529", "#6EB782", "tomato", "lightgrey"]

for encoding_id, encoding in enumerate(encodings):
    offset = width * multiplier
    rects = ax.bar(x + offset, gathered[:, encoding_id], width, label=encoding
                   , color=colors[encoding_id]
                   )
    # ax.bar_label(rects, padding=3)
    multiplier += 1

ax.spines[['right', 'top']].set_visible(False)
ax.set_xticks(x + width, datasets, rotation=65)
ax.set_yticks([i for i in np.arange(0, 1.1, .1)])
ax.set_xlim(-0.5, 22)
ax.legend(loc='lower center', ncols=len(encodings), frameon=False, bbox_to_anchor=(0.5, .92, 0.0, 0.0))
ax.set_ylabel('BAC')
plt.grid(ls=":", c=(.7, .7, .7))
plt.tight_layout()
plt.savefig("figures/comparison.png")
# plt.savefig("figures/comparison.eps")

"""
Statistical analysis
"""
alpha = .05
n_clfs = len(encodings)


# T-test
# ENCODING x DATASETS x FOLDS
print(ttest_data.shape)

t = []
for data_id, data in enumerate(datasets):
    # ENCODING x FOLDS
    data_scores = ttest_data[:, data_id, :]
    t.append(["%s" % data] + ["%.3f" % v for v in wilcoxon_data[data_id]])
    
    T, p = np.array(
                [[cv52cft(data_scores[i, :],
                        data_scores[j, :]) if i != j else (0.0, 1.0)
                    for i in range(n_clfs)]
                    for j in range(n_clfs)]
            ).swapaxes(0, 2)
    mean_adv = wilcoxon_data[data_id] < wilcoxon_data[data_id, :, np.newaxis]
    stat_adv = p < alpha
    
    _ = np.where(stat_adv * mean_adv)
    conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
    
    t.append([''] + [", ".join(["%i" % i for i in c])
        if len(c) > 0 and len(c) < n_clfs-1 else ("all" if len(c) == n_clfs-1 else "---")
        for c in conclusions])
    
tab_t = tabulate(t, floatfmt=".3f", headers=encodings, tablefmt="latex_booktabs")
print(tab_t)

# Wilcoxon
# DATASETS x ENCODINGS
print(wilcoxon_data.shape)
ranks = rankdata(wilcoxon_data, axis=1)
mean_ranks = np.mean(ranks, axis=0)

w = []
s = np.zeros((n_clfs, n_clfs))
p = np.zeros((n_clfs, n_clfs))

for i in range(n_clfs):
                for j in range(n_clfs):
                    s[i, j], p[i, j] = wilcoxon(wilcoxon_data.T[i], wilcoxon_data.T[j], zero_method="zsplit", alternative="greater")
                    
_ = np.where((p < alpha) * (s > 0))
conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
w.append([" "] + ["%.3f" % v for v in mean_ranks])

w.append([''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < n_clfs-1 else ("all" if len(c) == n_clfs-1 else "---")
                             for c in conclusions])

tab_w = tabulate(w, tablefmt="latex_booktabs")
print(tab_w)
print("Mean BAC:  ", np.mean(gathered, axis=0))