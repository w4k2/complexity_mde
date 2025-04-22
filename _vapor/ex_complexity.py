import numpy as np
import problexity as px
from utils import Data
from tqdm import tqdm


dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']

"""
Get results for each dataset
"""
scores_stml = np.load("results/ex00_preliminary_stml_transfer_224_scores.npy")
scores_igtd = np.load("results/ex00_preliminary_igtd_transfer_scores.npy")
scores_di = np.load("results/ex00_preliminary_di_transfer_224_scores.npy")
scores_pacman = np.load("results/ex00_preliminary_pacman_transfer_224_feature_scores_outline.npy")
# ENCODING x DATASETS x FOLDS x EPOCHS
gathered = np.stack((scores_stml, scores_igtd, scores_di, scores_pacman))
# Remove breastcan and bupa (2, 4)
del_datasets = list(np.delete(np.arange(gathered.shape[1]), [2, 4]))
# ENCODING x DATASETS x FOLDS
gathered = gathered[:, del_datasets, :, -1]
# ENCODING x DATASETS
mean_gathered = np.mean(gathered, axis=2)

np.save("results/complexity_mde_scores", mean_gathered)

"""
Calculate complexity for each dataset
"""
data = Data(selection=dataset_names, path="datasets/")
datasets = data.load()

complexity = []
for data_id, dataset_name in enumerate(tqdm(datasets)):
    X, y = datasets[dataset_name][0], datasets[dataset_name][1]
    cc = px.ComplexityCalculator()
    cc.fit(X, y)
    
    complexity.append(cc.complexity)
    
complexity = np.array(complexity)
np.save("results/complexity_measures", complexity)