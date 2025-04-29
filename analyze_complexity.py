import numpy as np
import problexity as px
import matplotlib.pyplot as plt
import matplotlib


dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']

complexity = np.load("results/complexity_measures.npy")
print(complexity.shape)

cc = px.ComplexityCalculator()
metrics = cc._metrics()

cmap = matplotlib.colormaps['tab20c']
colors = [cmap(i) for i in np.linspace(0, 1, len(metrics))]

for data_id, data_name in enumerate(dataset_names):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.bar(metrics, complexity[data_id], color=colors)
    ax.grid(ls=":", c=(.7, .7, .7))
    
    plt.tight_layout()
    plt.savefig("figures/complexity/%s_complexity.png" % (data_name))