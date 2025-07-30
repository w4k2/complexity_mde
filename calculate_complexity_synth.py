import numpy as np
import problexity as px
from utils import Data
from tqdm import tqdm


"""
Calculate complexity for each dataset
"""
datasets = np.load("synth_data/synth.npy")

complexity = []
for data_id, data in enumerate(tqdm(datasets)):
    X, y = data[:, :-1], data[:, -1].astype(int)
    cc = px.ComplexityCalculator()
    cc.fit(X, y)
    
    complexity.append(cc.complexity)
    
complexity = np.array(complexity)
np.save("results/complexity_measures_synth", complexity)