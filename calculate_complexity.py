import numpy as np
import problexity as px
from utils import Data
from tqdm import tqdm


dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']

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
np.save("results2/complexity_measures", complexity)