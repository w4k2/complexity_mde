import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.rcParams.update({'font.size': 22, "font.family" : "monospace"})


scores_stml = np.load("results/stream/gr_n_css999_rs1410_nd30_ln5_d50_250000_binary.npy")
scores_stml_rgb = np.load("results/stream/gr_n_css999_rs1410_nd30_ln5_d50_250000_rgb.npy")

fig, ax = plt.subplots(1, 1, figsize=(25, 10))

plt.plot(gaussian_filter1d(scores_stml, 2), color="dodgerblue", label="STML")
plt.plot(gaussian_filter1d(scores_stml_rgb, 2), color="tomato", label="STML RGB")

ax.set_xlabel("data chunk")
ax.set_ylabel("BAC")
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylim(0.7, 1)
ax.set_xlim(-10, 1000)
plt.grid(ls=":", c=(.7, .7, .7))
plt.legend(frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig("figures/stream.png", dpi=200)



