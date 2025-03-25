import numpy as np
import matplotlib.pyplot as plt
from utils import Data
from scipy.stats import rankdata
# Stat
from utils import cv52cft
from scipy.stats import rankdata, wilcoxon
from tabulate import tabulate
from time import sleep
import matplotlib
matplotlib.rcParams.update({'font.size': 22, "font.family" : "monospace"})

n_data = 14
data = Data(selection=("all", ["balanced", "binary"]), path="datasets/")
datasets = data.load()
# datasets = list(datasets.keys())[:n_data]
datasets = ["breastcancoimbra"]
encodings = ["STML", "STML_RGB"]

# scores_stml = np.load("results/ex00_preliminary_stml_transfer_224_scores.npy")
# scores_stml_rgb = np.load("results/ex00_preliminary_stml_rgb_transfer_224_scores.npy")

scores_stml = np.load("results/ex00_breastcancoimbra_epochs_stml.npy")
scores_stml_rgb = np.load("results/ex00_breastcancoimbra_epochs_stml_rgb.npy")

# ENCODING x DATASETS x FOLDS x EPOCHS
gathered = np.stack((scores_stml, scores_stml_rgb))
# ENCODING x DATASETS x EPOCHS
gathered = np.mean(gathered, axis=2)

colors = ["lightsteelblue", "tomato", "lightgrey"]

for data_id, dataset in enumerate(datasets):
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.plot(gathered[0, data_id])
    ax.plot(gathered[1, data_id])

    ax.set_ylim(.5, 1)
    # ax.set_xlim(0, 20)
    plt.title(dataset)
    plt.grid(ls=":", c=(.7, .7, .7))
    plt.tight_layout()
    plt.savefig("bar.png")
    plt.close()
    sleep(2)
    # exit()
# plt.savefig("figures/epochs.png")
# plt.savefig("figures/comparison.eps")