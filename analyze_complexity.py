import numpy as np
import problexity as px
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter1d
matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


cc = px.ComplexityCalculator()
metrics = cc._metrics()

dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']

complexity = np.load("results2/complexity_measures.npy")
print(np.mean(complexity, axis=1))
     
"""
Complexity space PCA
"""
# pca = PCA(n_components=2, random_state=None)
# complexity_space = pca.fit_transform(complexity)
# print(pca.explained_variance_ratio_)

# cmap = matplotlib.colormaps['tab20c']
# colors = [cmap(i) for i in np.linspace(0, 1, len(dataset_names)+1)]

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# for data_id, data_name in enumerate(dataset_names):
#     ax.scatter(complexity_space[data_id, 0], complexity_space[data_id, 1], color=colors[data_id+1])
#     ax.text(complexity_space[data_id, 0]+.03, complexity_space[data_id, 1]-.02, s=dataset_names[data_id], fontsize=8)


"""
Complexity signature
"""
# best_data = ["heart", "breastcancoimbra", "pima", "liver", "german", "australian", "mammographic", "spambase", "banknote"]
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# for data_id, data_name in enumerate(dataset_names):
#     if data_name in best_data:
#         ax.plot(gaussian_filter1d(complexity[data_id], 4), color=colors[data_id+1])
#         ax.plot(gaussian_filter1d(complexity[data_id], 4), color=colors[data_id+1])

# ax.set_xticks(np.arange(0, 22, 1), metrics, rotation=45)
# ax.set_xlim(0, 21)
# ax.grid(ls=":", c=(.7, .7, .7))
# plt.tight_layout()
# plt.savefig("figures/complexity/all_signature.png")
# exit()

"""
Metric values for each dataset
"""

cmap = matplotlib.colormaps['tab20b']
colors = np.array([cmap(i) for i in np.linspace(0, 1, len(metrics))])

for data_id, data_name in enumerate(dataset_names):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.bar(metrics, complexity[data_id], color=colors)
    ax.grid(ls=":", c=(.7, .7, .7))
    ax.set_xticks(np.arange(0, 22, 1), metrics, rotation=45)
    ax.set_title(data_name)
    ax.set_ylabel("Complexity measure value")
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("figures2/complexity2/%s_complexity.png" % (data_name))
    plt.savefig("figures2/complexity2/%s_complexity.eps" % (data_name))