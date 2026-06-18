import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from scipy.ndimage import gaussian_filter1d
from tabulate import tabulate
from scipy import stats
from scipy.stats import rankdata, wilcoxon
import matplotlib
matplotlib.rcParams.update({'font.size': 11, "font.family" : "monospace"})

dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']

# dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']

# DATASETS x FOLDS x TRANSFER (imagenet | best BAC | best transrate) x EPOCH
# to do after exp_comparison
# imgnet
# scores = np.load("results/transfer/comparison_imgnet_finetuning.npy")
# scores_complexity = np.load("results/transfer/comparison_imgnet_finetuning_complexity.npy")

# wo
scores = np.load("results/transfer/comparison_wo_imgnet_finetuning.npy")
scores_complexity = np.load("results/transfer/comparison_wo_imgnet_finetuning_complexity.npy")
scores = scores[:14]
scores_complexity = scores_complexity[:14]

# remove the 5th dataset
# scores = np.delete(scores, 4, axis=0)
# scores_complexity = np.delete(scores_complexity, 4, axis=0)

# DATASETS x TRANSFER (imagenet | best BAC | best transrate) x EPOCH
mean_scores = np.mean(scores, axis=1)
mean_scores_complexity = np.mean(scores_complexity, axis=1)

transfer_names = ["Imagenet", "Best BAC", "Best TransRate"]

# for each dataset
# for data_id in range(mean_scores.shape[0]):
#     fig, ax = plt.subplots(1, 1, figsize=(15, 10))

#     for i, scores in enumerate(mean_scores[data_id]):
#         ax.plot(gaussian_filter1d(scores, 1), label=transfer_names[i])

#     ax.plot(gaussian_filter1d(mean_scores_complexity[data_id, 0], 3), label="Highest complexity")
    
#     ax.set_xticks(np.arange(0, 50, 1), [str(i+1) for i in np.arange(0, 50, 1)])
#     ax.set_xlim(-.2, 49.2)
#     ax.grid(ls=":", c=(.7, .7, .7))
#     ax.set_title(dataset_names[data_id])

#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("foo.png")
#     # sleep(2)
#     plt.close()

# mean plot for all datasets
# TRANSFER (imagenet | best BAC | best transrate) x EPOCH
data_mean_scores = np.mean(mean_scores, axis=0)
data_mean_scores_complexity = np.mean(mean_scores_complexity, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(11, 5))

for i, scores in enumerate(data_mean_scores):
    if i == 0:
        ax.plot(gaussian_filter1d(scores, 1), label="Imagenet")
    else:
        ax.plot(gaussian_filter1d(scores, 1), label=transfer_names[i])
ax.plot(gaussian_filter1d(data_mean_scores_complexity[0], 1), label="Highest complexity")

ax.set_xticks(np.arange(0, 50, 1), [str(i+1) for i in np.arange(0, 50, 1)])
ax.set_xlim(-.2, 49.2)
ax.set_ylim(.45, 1.0)
ax.grid(ls=":", c=(.7, .7, .7))
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel("Mean Balanced accuracy score over all datasets")
ax.set_xlabel("Fine-tuning Epoch")
# ax.set_title("Fine-tuning previously selected models (trained from scratch)")
ax.set_title("Fine-tuning previously selected models (with fine-tuned ImageNet weights)")

plt.legend()
plt.tight_layout()
plt.savefig("bar.png")
# plt.savefig("figures/comparison_wo_imgnet.eps")
plt.savefig("figures/comparison_imgnet.eps")
plt.close()

# Table
def cv52cft(a, b):
    d = a.reshape(2, 5) - b.reshape(2, 5)
    f = np.sum(np.power(d, 2)) / (2 * np.sum(np.var(d, axis=0, ddof=0)))
    p = 1-stats.f.cdf(f, 10, 5)
    return f, p

def t_test_corrected(a, b, J=2, k=5):
    """
    Corrected t-test for repeated cross-validation.
    input, two 2d arrays. Repetitions x folds
    As default for 5x5CV
    """
    if J*k != a.shape[0]:
        raise Exception('%i scores received, but J=%i, k=%i (J*k=%i)' % (
            a.shape[0], J, k, J*k
        ))

    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (J * k) + 1 / (k - 1)) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), (k * J) - 1) * 2
    return t_stat, pval



transfer_names = ["Imagenet", "Best BAC", "Best TransRate", "Highest complexity"]

# scores = np.load("results/transfer/comparison_imgnet_finetuning.npy")
# scores_complexity = np.load("results/transfer/comparison_imgnet_finetuning_complexity.npy")

scores = np.load("results/transfer/comparison_wo_imgnet_finetuning.npy")
scores_complexity = np.load("results/transfer/comparison_wo_imgnet_finetuning_complexity.npy")

# remove the 5th dataset
# scores = np.delete(scores, 4, axis=0)
# scores_complexity = np.delete(scores_complexity, 4, axis=0)

# for i, data in  enumerate(scores):
#     if np.mean(data[:, 2]) == 0:
#         data[:, 2] = data[:, 1]
# exit()

# DATA x FOLDS x MODEL x EPOCHS
total_scores = np.concatenate((scores, scores_complexity), axis=2)

# not all datasets
# total_scores = total_scores[:14]
# dataset_names = dataset_names[:14]

# DATA x FOLDS x MODEL (choose epoch)
total_scores = total_scores[:, :, :, 49]
# DATA x MODEL
mean_total_scores = np.mean(total_scores, axis=1)
# print(data_mean_scores)

print(np.mean(mean_total_scores, axis=0))

t = []
alpha = .05
for data_id, data in enumerate(dataset_names):
    t.append(["%s" % data] + ["%.3f" % v for v in mean_total_scores[data_id]])
    # FOLDS x MODEL (choose epoch)
    data_total_scores = total_scores[data_id]
    
    T, p = np.array(
                [[t_test_corrected(data_total_scores[:, i],
                        data_total_scores[:, j]) if i != j else (0.0, 1.0)
                    for i in range(len(transfer_names))]
                    for j in range(len(transfer_names))]
            ).swapaxes(0, 2)
    mean_adv = mean_total_scores[data_id] < mean_total_scores[data_id, :, np.newaxis]
    stat_adv = p < alpha
    
    _ = np.where(stat_adv * mean_adv)
    conclusions = [list(1 + _[1][_[0] == i]) for i in range(len(transfer_names))]
    
    t.append([''] + [", ".join(["%i" % i for i in c])
        if len(c) > 0 and len(c) < len(transfer_names)-1 else ("all" if len(c) == len(transfer_names)-1 else "---")
        for c in conclusions])
    
    
print(tabulate(t, headers=transfer_names, floatfmt="%.3f", tablefmt="latex_booktabs"))

# Wilcoxon
# DATASETS x ENCODINGS
ranks = rankdata(mean_total_scores, axis=1)
print(np.mean(mean_total_scores, axis=0))
mean_ranks = np.mean(ranks, axis=0)

w = []
s = np.zeros((4, 4))
p = np.zeros((4, 4))

for i in range(4):
                for j in range(4):
                    s[i, j], p[i, j] = wilcoxon(mean_total_scores.T[i], mean_total_scores.T[j], zero_method="zsplit", alternative="greater")
                    
_ = np.where((p < alpha) * (s > 0))
conclusions = [list(1 + _[1][_[0] == i]) for i in range(4)]
w.append([" "] + ["%.3f" % v for v in mean_ranks])

w.append([''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < 4-1 else ("all" if len(c) == 4-1 else "---")
                             for c in conclusions])

tab_w = tabulate(w, headers=transfer_names, tablefmt="latex_booktabs")
print(tab_w)
