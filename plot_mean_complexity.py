import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18.5, "font.family" : "monospace"})


dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']

comp = np.load("results/complexity_measures.npy")
mean_comp = np.mean(comp, axis=1)
print(mean_comp.shape)

fig, ax = plt.subplots(1, 1, figsize=(15, 8))

cmap = matplotlib.colormaps['tab20c']
colors = [cmap(i) for i in np.linspace(0, 1, len(dataset_names)+1)]

ax.bar(dataset_names, mean_comp, color=colors[1:])

ax.set_xticks(np.arange(0, 20, 1), dataset_names, rotation=45)
ax.grid(ls=":", c=(.7, .7, .7))
ax.set_ylim((0, .6))
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel("Mean complexity over 22 measures")

plt.tight_layout()
plt.savefig("figures/complexity/mean_complexity.png")
plt.savefig("figures/complexity/mean_complexity.eps")