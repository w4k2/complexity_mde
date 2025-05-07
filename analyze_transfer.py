import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import problexity as px
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats.stats import pearsonr
matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


# dataset_names = ['australian', 'banknote', 'breastcan', 'breastcancoimbra', 'bupa', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']
# Remove breastcan and bupa (2, 4)
dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']
# dataset_names = ['australian', "banknote"]
# transfer_names = ["imagenet"] + dataset_names + ["synth"]
transfer_names = ["imagenet"] + ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin'] 
# + ["gpt"]

# DATASET x FOLDS x TRANSFER
scores = np.load("results/transfer/di_bac_full.npy")
transrates = np.load("results/transfer/di_transrates_full.npy")
# scores = np.load("results/transfer/di_bac_full_wo_imgnet.npy")
# transrates = np.load("results/transfer/di_transrates_full_wo_imgnet.npy")

# # Remove breastcan and bupa (2, 4)
del_datasets = list(np.delete(np.arange(scores.shape[0]), [2, 4]))
del_transfer = list(np.delete(np.arange(scores.shape[2]), [3, 5]))
scores = scores[del_datasets]
transrates = transrates[del_datasets]

# """
# Add GPT
# """
# best
# gpt_scores = np.load("results/transfer/di_bac_synth15260True_imgnet.npy")
# gpt_transrates = np.load("results/transfer/di_transrates_synth15260True_imgnet.npy")
# worse
# gpt_scores = np.load("results/transfer/di_bac_synth1960True_imgnet.npy")
# gpt_transrates = np.load("results/transfer/di_transrates_synth1960True_imgnet.npy")

# scores = np.concatenate((scores, gpt_scores), axis=2)
# transrates = np.concatenate((transrates, gpt_transrates), axis=2)

print(scores.shape)
print(transrates.shape)
# print(gpt_scores.shape)
# print(gpt_transrates.shape)

# Calculate mean across folds
mean_scores = np.mean(scores, axis=1)
mean_transrates = np.mean(transrates, axis=1)
"""
mean scatter plot
"""
# Calculate mean withou bac and transfer for the same dataset
mean_plot_scores = []
mean_plot_transrates = []

all_idx = np.arange(0, 20, 1)

for i in range(len(transfer_names)):
    if i == 0 or i == 1:
        _ = np.mean(mean_scores[:, i])
        mean_plot_scores.append(_)
        _ = np.mean(mean_transrates[:, i])
        mean_plot_transrates.append(_)
    else:
        calc_idx = np.delete(all_idx, i-1)
        _ = np.mean(mean_scores[calc_idx, i])
        mean_plot_scores.append(_)
        _ = np.mean(mean_transrates[calc_idx, i])
        mean_plot_transrates.append(_)

mean_plot_scores = np.array(mean_plot_scores)
mean_plot_transrates = np.array(mean_plot_transrates)

# Complexity x BAC plot
complexity = np.load("results/complexity_measures.npy")

# Best metrics for trasnferability estimation
cc = px.ComplexityCalculator()
metrics = cc._metrics()
print(metrics)
kbest = SelectKBest(f_regression)
kbest.fit_transform(complexity, mean_plot_scores[1:])
kbest_scores = kbest.scores_
kbest_argmax = np.argsort(-kbest_scores)
print(kbest_scores[kbest_argmax[-1:]])
print(np.array(metrics)[kbest_argmax])



"""
Plot f_regression
"""
matplotlib.rcParams.update({'font.size': 20, "font.family" : "monospace"})
fig, ax = plt.subplots(1, 1, figsize=(8, 12))

cmap = matplotlib.colormaps['tab20b']
colors = np.array([cmap(i) for i in np.linspace(0, 1, len(metrics))])[np.flip(kbest_argmax)]

ax.barh(np.array(metrics)[np.flip(kbest_argmax)], kbest_scores[np.flip(kbest_argmax)], color=colors)

ax.grid(ls=":", c=(.7, .7, .7))
ax.set_title("ResNet-18 trained from scratch")
# ax.set_title("Pre-trained on ImageNet", fontsize=24)
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel("F-statistic value")
ax.set_xscale('log')

plt.tight_layout()
plt.savefig("figures/complexity/f_regression_wo_imgnet.png")
plt.savefig("figures/complexity/f_regression_wo_imgnet.eps")
# plt.savefig("figures/complexity/f_regression_imgnet.png")
# plt.savefig("figures/complexity/f_regression_imgnet.eps")
matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

# exit()

# <eamcomplexity taking into account selected metrics
# mean_complexity = np.mean(complexity[:, kbest_argmax[-1:]], axis=1)
# mean_complexity = np.mean(complexity[:, kbest_argmax[:1]], axis=1)
mean_complexity = complexity[:, kbest_argmax][:, 0]
# mean_complexity = np.mean(complexity[:, kbest_argmax][:, [0, 1, 2, 3]], axis=1)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
cmap = matplotlib.colormaps['tab20c']
colors = [cmap(i) for i in np.linspace(0, 1, len(dataset_names)+1)]

for data_id in range(len(dataset_names)):
    ax.scatter(mean_plot_scores[data_id+1], mean_complexity[data_id], color=colors[data_id+1], s=80)
    ax.text(mean_plot_scores[data_id+1]-.003, mean_complexity[data_id]+.005, s=dataset_names[data_id], fontsize=12)
    # ax.text(mean_plot_scores[data_id+1]-.001, mean_complexity[data_id]+.005, s=dataset_names[data_id], fontsize=10, rotation=-45)

# ax.set_title("ResNet-18 trained from scratch \n Pearson correlation coefficient: %.3f, p-value: %.3f" % (pearsonr(mean_complexity, mean_plot_scores[1:])[0], pearsonr(mean_complexity, mean_plot_scores[1:])[1]))
ax.set_title("Fine-tuned ResNet-18 pre-trained on ImageNet \n Pearson correlation coefficient: %.3f, p-value: %.3f "% (pearsonr(mean_complexity, mean_plot_scores[1:])[0], pearsonr(mean_complexity, mean_plot_scores[1:])[1]))
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel("Mean Balanced accuracy")
# ax.set_ylabel("L2")
ax.set_ylabel("N1")
ax.grid(ls=":", c=(.7, .7, .7))

# wo
# ax.set_xlim((.66, 0.7))
# imgnet
ax.set_xlim((.74, 0.8))
ax.set_ylim((-0.005, .25))

plt.tight_layout()
# wo
# plt.savefig("figures/all_bac_complexity_wo_imgnet.png", dpi=200)
# plt.savefig("figures/all_bac_complexity_wo_imgnet.eps")
# imgnet
plt.savefig("figures/all_bac_complexity.png", dpi=200)
plt.savefig("figures/all_bac_complexity.eps")
plt.close()
print(f_regression(mean_complexity.reshape(-1, 1), mean_plot_scores[1:]))
print(pearsonr(mean_complexity, mean_plot_scores[1:]))
# exit()



fig, ax = plt.subplots(1, 1, figsize=(10, 10))
cmap = matplotlib.colormaps['tab20c']
colors = [cmap(i) for i in np.linspace(0, 1, len(transfer_names))]

for transfer_id in range(len(transfer_names)):
    ax.scatter(mean_plot_scores[transfer_id], mean_plot_transrates[transfer_id], color=colors[transfer_id], s=80)
    # ax.text(mean_plot_scores[transfer_id]+.0003, mean_plot_transrates[transfer_id]-.005, s=transfer_names[transfer_id], fontsize=10, rotation=-45)
    # wo
    # if transfer_names[transfer_id] == "imagenet":
    #     transfer_names[transfer_id] = "random"
    # ax.text(mean_plot_scores[transfer_id]-.001, mean_plot_transrates[transfer_id]+.0015, s=transfer_names[transfer_id], fontsize=12, rotation=0)
    # imgnet
    ax.text(mean_plot_scores[transfer_id]-.001, mean_plot_transrates[transfer_id]+.006, s=transfer_names[transfer_id], fontsize=12, rotation=90)

# ax.scatter(np.mean(gpt_scores, axis=(0, 1)), np.mean(gpt_transrates, axis=(0, 1)), color="tomato", s=80)

# ax.set_title("Mean plot over all transfer datasets")
# ax.set_title("ResNet-18 trained from scratch \n Pearson correlation coefficient: %.3f, p-value: %.3f" % (pearsonr(mean_plot_transrates[1:], mean_plot_scores[1:])[0], pearsonr(mean_plot_transrates[1:], mean_plot_scores[1:])[1]))
ax.set_title("Fine-tuned ResNet-18 pre-trained on ImageNet \n Pearson correlation coefficient: %.3f, p-value: %.3f "% (pearsonr(mean_plot_transrates[1:], mean_plot_scores[1:])[0], pearsonr(mean_plot_transrates[1:], mean_plot_scores[1:])[1]))
ax.set_xlabel("Mean Balanced accuracy")
ax.set_ylabel("Mean TransRate")
ax.spines[['right', 'top']].set_visible(False)
# wo
# ax.set_xlim((.66, 0.7))
# ax.set_ylim((-.065, .08))
# imgnet
ax.set_xlim((.74, 0.8))
ax.set_ylim((-.255, .05))
ax.grid(ls=":", c=(.7, .7, .7))
# plt.legend(frameon=True, fontsize=12, ncols=2)

plt.tight_layout()
# wo
# plt.savefig("figures/transfer/all_transfer_wo_imgnet.png", dpi=200)
# plt.savefig("figures/transfer/all_transfer_wo_imgnet.eps")
# imgnet
plt.savefig("figures/transfer/all_transfer.png", dpi=200)
plt.savefig("figures/transfer/all_transfer.eps")

plt.close()
print(f_regression(mean_plot_transrates[1:].reshape(-1, 1), mean_plot_scores[1:]))
print(pearsonr(mean_plot_transrates[1:], mean_plot_scores[1:]))
exit()





"""
Heatmaps for TransRate and BAC
"""
fig, ax = plt.subplots(1, 2, figsize=(40, 20))

for i in range(len(dataset_names)):
    for j in range(len(transfer_names)):
        if i+1 == j:
            mean_scores[i, j] = np.median(mean_scores)
            mean_transrates[i, j] = np.median(mean_transrates)

ax[0].imshow(mean_scores, cmap="binary_r")
ax[1].imshow(mean_transrates, cmap="binary_r")

ax[0].set_xticks(range(len(transfer_names)), labels=transfer_names, rotation=65, fontsize=18)
ax[0].set_yticks(range(len(dataset_names)), labels=dataset_names, fontsize=18)

ax[1].set_xticks(range(len(transfer_names)), labels=transfer_names, rotation=65, fontsize=18)
ax[1].set_yticks(range(len(dataset_names)), labels=dataset_names, fontsize=18)

ax[0].set_title("Balanced accuracy score", fontsize=25)
ax[1].set_title("TransRate", fontsize=25)

for i in range(len(dataset_names)):
    for j in range(len(transfer_names)):
        if i+1 != j:
            if np.round(mean_scores[i, j], 2) > np.median(mean_scores):
                text = ax[0].text(j, i, np.round(mean_scores[i, j], 2),
                        ha="center", va="center", color="black")
            else:
                text = ax[0].text(j, i, np.round(mean_scores[i, j], 2),
                        ha="center", va="center", color="w")
                
            if np.round(mean_transrates[i, j], 2) > np.median(mean_transrates):
                text = ax[1].text(j, i, np.round(mean_transrates[i, j], 2),
                        ha="center", va="center", color="black")
            else:
                text = ax[1].text(j, i, np.round(mean_transrates[i, j], 2),
                        ha="center", va="center", color="w")
        else:
            text = ax[0].text(j, i, "—",
                        ha="center", va="center", color="white")
            text = ax[1].text(j, i, "—",
                        ha="center", va="center", color="white")
            


fig.tight_layout()
plt.savefig("figures/transfer/heatmap_wo_imgnet.png", dpi=200)
plt.savefig("figures/transfer/heatmap_wo_imgnet.eps")
exit()




"""
For each dataset
"""
for data_id, data_name in enumerate(dataset_names):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    cmap = matplotlib.colormaps['tab20c']

    colors = [cmap(i) for i in np.linspace(0, 1, len(transfer_names))]

    for transfer_id, transfer in enumerate(transfer_names):
        if transfer != data_name:
            transfer
            ax.scatter(mean_scores[data_id][transfer_id], mean_transrates[data_id][transfer_id], label=transfer, color=colors[transfer_id])
            ax.text(mean_scores[data_id][transfer_id]+.001, mean_transrates[data_id][transfer_id]-.01, s=transfer, fontsize=8)
            
        # if transfer != data_name and transfer != 'imagenet':
        #     # complexity
        #     ax.scatter(mean_scores[data_id][transfer_id], mean_complexity[transfer_id-1], color=colors[transfer_id])
        #     ax.text(mean_scores[data_id][transfer_id]+.001, mean_complexity[transfer_id-1]-.01, s=transfer, fontsize=8)

    # ax.scatter(np.mean(gpt_scores, axis=(1))[data_id], np.mean(gpt_transrates, axis=(1))[data_id], color="tomato", s=80)

    ax.set_title(data_name)
    ax.set_xlabel("BAC")
    ax.set_ylabel("TransRate")
    # ax.set_xticks(np.arange(.55, .63, .01))
    # ax.set_yticks(np.arange(-.8, 1, .1))
    # ax.set_xlim((.55, .61))
    # ax.set_ylim((-.8, .9))
    ax.grid(ls=":", c=(.7, .7, .7))
    # plt.legend(frameon=True, fontsize=12, ncols=2)

    plt.tight_layout()
    plt.savefig("figures/transfer/di_%s" % data_name, dpi=200)
    plt.close()