import numpy as np
import problexity as px


encodings = ["STML", "IGTD", "DI", "RETIRE"]
dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']
cc = px.ComplexityCalculator()
complexity_metrics = cc._metrics()

# ENCODING x DATASETS
scores = np.load("results/complexity_mde_scores.npy")
# DATASETS x COMPLEXITY METRICS
complexity = np.load("results/complexity_measures.npy")

# Meta dataset
meta = []
for data_id, data_name in enumerate(dataset_names):
    for i in range(len(encodings)):
        meta_ = np.zeros((1 + complexity.shape[1] + 1 + 1)).astype("U")
        meta_[0] = data_name
        meta_[1:23] = complexity[data_id]
        meta_[23] = i
        meta_[24] = scores[data_id, i]
        meta.append(list(meta_))
meta = np.array(meta)
print(meta)
np.save("results/meta_dataset", meta)