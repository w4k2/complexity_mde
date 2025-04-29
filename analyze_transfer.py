import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18, "font.family" : "monospace"})


# dataset_names = ['australian', 'banknote', 'breastcan', 'breastcancoimbra', 'bupa', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']
# Remove breastcan and bupa (2, 4)
dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']
# dataset_names = ['australian', "banknote"]
# transfer_names = ["imagenet"] + dataset_names + ["synth"]
transfer_names = ["imagenet"] + ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']

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
    if i == 0:
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

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
cmap = matplotlib.colormaps['tab20c']
colors = [cmap(i) for i in np.linspace(0, 1, len(transfer_names))]

for transfer_id in range(len(transfer_names)):
    ax.scatter(mean_plot_scores[transfer_id], mean_plot_transrates[transfer_id], color=colors[transfer_id])
    ax.text(mean_plot_scores[transfer_id]+.0005, mean_plot_transrates[transfer_id]-.005, s=transfer_names[transfer_id], fontsize=8)

ax.set_title("Mean plot over all transfer datasets")
ax.set_xlabel("BAC")
ax.set_ylabel("TransRate")
# ax.set_xticks(np.arange(.55, .63, .01))
# ax.set_yticks(np.arange(-.8, 1, .1))
# ax.set_xlim((.55, .61))
# ax.set_ylim((-.8, .9))
ax.grid(ls=":", c=(.7, .7, .7))
# plt.legend(frameon=True, fontsize=12, ncols=2)

plt.tight_layout()
plt.savefig("figures/transfer/full.png", dpi=200)
plt.close()
exit()


"""
Heatmaps for TransRate and BAC
"""
fig, ax = plt.subplots(1, 2, figsize=(40, 20))
ax[0].imshow(mean_scores, cmap="winter")
ax[1].imshow(mean_transrates, cmap="winter")

ax[0].set_xticks(range(len(transfer_names)), labels=transfer_names, rotation=65)
ax[0].set_yticks(range(len(dataset_names)), labels=dataset_names)

ax[1].set_xticks(range(len(transfer_names)), labels=transfer_names, rotation=65)
ax[1].set_yticks(range(len(dataset_names)), labels=dataset_names)

ax[0].set_title("BAC")
ax[1].set_title("TransRate")

for i in range(len(dataset_names)):
    for j in range(len(transfer_names)):
        text = ax[0].text(j, i, np.round(mean_scores[i, j], 2),
                       ha="center", va="center", color="w")
        text = ax[1].text(j, i, np.round(mean_transrates[i, j], 2),
                       ha="center", va="center", color="w")


fig.tight_layout()
plt.savefig("figures/transfer/heatmap.png", dpi=200)

"""
For each dataset
"""
for data_id, data_name in enumerate(dataset_names):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    cmap = matplotlib.colormaps['tab20c']

    colors = [cmap(i) for i in np.linspace(0, 1, len(transfer_names))]

    for transfer_id, transfer in enumerate(transfer_names):
        if transfer != data_name:
            ax.scatter(mean_scores[data_id][transfer_id], mean_transrates[data_id][transfer_id], label=transfer, color=colors[transfer_id])
            ax.text(mean_scores[data_id][transfer_id]+.001, mean_transrates[data_id][transfer_id]-.01, s=transfer, fontsize=8)

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