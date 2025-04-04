import numpy as np



scores_stml = np.load("results/ex00_preliminary_stml_transfer_224_scores.npy")
mean_scores_stml = np.mean(scores_stml, axis=1)
print(mean_scores_stml[0].mean())

scores_ensemble = np.load("results/channel_ensemble.npy")
mean_scores_ensemble = np.mean(scores_ensemble, axis=1)
print(mean_scores_ensemble[0].mean())