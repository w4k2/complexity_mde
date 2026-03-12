import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 14, "font.family" : "monospace"})

RUNTIME_CSV = "results_runtime/runtime_summary_per_fold_4.csv"
OUT_DIR = "figures_runtime_with_extraction"
os.makedirs(OUT_DIR, exist_ok=True)

runtime_df = pd.read_csv(RUNTIME_CSV)
runtime_df["mean_seconds"] = runtime_df["mean_seconds"].astype(float)

def get_mean_std(method, init_mode):
    vals = runtime_df[
        (runtime_df["method"] == method) &
        (runtime_df["init"] == init_mode)
    ]["mean_seconds"].values
    return float(np.mean(vals)), float(np.std(vals))


def get_vals(method, init_mode):
    return runtime_df[
        (runtime_df["method"] == method) &
        (runtime_df["init"] == init_mode)
    ]["mean_seconds"].values


def get_mean_std(method, init_mode):
    vals = get_vals(method, init_mode)
    return float(np.mean(vals)), float(np.std(vals))


complexity_mean, complexity_std = get_mean_std("complexity_22", "raw_tabular")
encoding_mean, encoding_std = get_mean_std("encoding", "shared")

feat_img_mean, feat_img_std = get_mean_std("feature_extraction", "imagenet")
feat_rnd_mean, feat_rnd_std = get_mean_std("feature_extraction", "random")

hs_img_mean, hs_img_std = get_mean_std("hscore", "imagenet")
hs_rnd_mean, hs_rnd_std = get_mean_std("hscore", "random")

tr_img_mean, tr_img_std = get_mean_std("transrate", "imagenet")
tr_rnd_mean, tr_rnd_std = get_mean_std("transrate", "random")

n2_mean, n2_std = get_mean_std("complexity_n2", "raw_tabular")
l1_mean, l1_std = get_mean_std("complexity_l1", "raw_tabular")

# totals
hs_img_total = encoding_mean + feat_img_mean + hs_img_mean
hs_rnd_total = encoding_mean + feat_rnd_mean + hs_rnd_mean
tr_img_total = encoding_mean + feat_img_mean + tr_img_mean
tr_rnd_total = encoding_mean + feat_rnd_mean + tr_rnd_mean

# 1
labels = [
    "Complexity\n(22)",
    "H-score\nImageNet",
    "TransRate\nImageNet",
    "N2\nRandom",
    "L1\nImageNet",
]

x = np.arange(len(labels))
w = 0.72

encoding_part = np.array([0.0, encoding_mean, encoding_mean, 0.0, 0.0,])
feature_part = np.array([0.0, feat_img_mean, feat_img_mean, 0.0, 0.0,])
metric_part = np.array([complexity_mean, hs_img_mean, tr_img_mean, n2_mean, l1_mean,])
totals = encoding_part + feature_part + metric_part
fig, ax = plt.subplots(figsize=(12,7))

ax.bar(x, encoding_part, width=w, color="#4C78A8", label="Encoding")
ax.bar(x, feature_part, width=w, bottom=encoding_part, color="#72B7B2", label="Feature extraction")
ax.bar(x, metric_part, width=w, bottom=encoding_part + feature_part, color="#F58518", label="Data transferability measures")

for i in range(len(x)):
    if encoding_part[i] > 0:
        ax.text(x[i], encoding_part[i] / 2, f"{encoding_part[i]:.3f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    if feature_part[i] > 0:
        ax.text(x[i], encoding_part[i] + feature_part[i] / 2, f"{feature_part[i]:.3f}", ha="center", va="center", color="black", fontsize=9)

    ax.text(x[i], encoding_part[i] + feature_part[i] + metric_part[i] / 2, f"{metric_part[i]:.3f}", ha="center", va="center", color="black", fontsize=9)
    ax.text(x[i], totals[i] + max(totals) * 0.02, f"Σ={totals[i]:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x, labels)
ax.set_ylabel("Mean runtime [s]")
ax.set_title("Runtime decomposition: encoding + extraction + metric")
ax.grid(ls=":", c=(.7, .7, .7), axis="y")
ax.spines[['right', 'top']].set_visible(False)
ax.legend(frameon=True)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/runtime_stacked_full_pipeline_custom.png", dpi=220)
plt.savefig(f"{OUT_DIR}/runtime_stacked_full_pipeline_custom.eps")
plt.close()


# 2
metric_only = np.array([hs_img_mean, hs_rnd_mean, tr_img_mean, tr_rnd_mean])
full_pipeline = np.array([hs_img_total, hs_rnd_total, tr_img_total, tr_rnd_total])
labels_transfer = [
    "H-score\nImageNet",
    "H-score\nRandom",
    "TransRate\nImageNet",
    "TransRate\nRandom",
]
x2 = np.arange(len(labels_transfer))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x2 - 0.18, metric_only, width=0.36, label="Metric only", color="#F58518")
ax.bar(x2 + 0.18, full_pipeline, width=0.36, label="Encoding + extraction + metric", color="#4C78A8")

ax.set_xticks(x2, labels_transfer)
ax.set_ylabel("Mean runtime [s]")
ax.set_title("Metric-only vs full pipeline runtime")
ax.grid(ls=":", c=(.7, .7, .7), axis="y")
ax.spines[['right', 'top']].set_visible(False)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/runtime_metric_vs_full_pipeline.png", dpi=220)
plt.savefig(f"{OUT_DIR}/runtime_metric_vs_full_pipeline.eps")
plt.close()

# 3
def dataset_mean(method, init_mode):
    return (
        runtime_df[
            (runtime_df["method"] == method) &
            (runtime_df["init"] == init_mode)
        ]
        .groupby("dataset")["mean_seconds"]
        .mean()
    )

complexity22_ds = dataset_mean("complexity_22", "raw_tabular")
encoding_ds = dataset_mean("encoding", "shared")
feat_img_ds = dataset_mean("feature_extraction", "imagenet")
hs_img_ds = dataset_mean("hscore", "imagenet")
tr_img_ds = dataset_mean("transrate", "imagenet")

hs_full = encoding_ds + feat_img_ds + hs_img_ds
tr_full = encoding_ds + feat_img_ds + tr_img_ds

df = pd.DataFrame({
    "complexity_22": complexity22_ds,
    "hscore_full": hs_full,
    "transrate_full": tr_full,
}).reset_index()

plot_df = df.sort_values("complexity_22", ascending=False).reset_index(drop=True)

x = np.arange(len(plot_df))
w = 0.26

fig, ax = plt.subplots(figsize=(16, 7))

ax.bar(x - w, plot_df["complexity_22"], width=w, label="Complexity (22)")
ax.bar(x,     plot_df["hscore_full"],  width=w, label="H-score")
ax.bar(x + w, plot_df["transrate_full"], width=w, label="TransRate")

ax.set_xticks(x, plot_df["dataset"], rotation=45, ha="right")
ax.set_ylabel("Mean runtime [s]")
ax.set_title("Dataset-level runtime comparison")
ax.grid(ls=":", c=(.7, .7, .7), axis="y")
ax.spines[['right', 'top']].set_visible(False)
ax.legend(frameon=True)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/runtime_dataset_comparison_main.png", dpi=220)
plt.savefig(f"{OUT_DIR}/runtime_dataset_comparison_main.eps")
plt.close()

print("Saved plots to:", OUT_DIR)

complexity_n2_ds = dataset_mean("complexity_n2", "raw_tabular")
complexity_l1_ds = dataset_mean("complexity_l1", "raw_tabular")

df2 = pd.DataFrame({
    "n2": complexity_n2_ds,
    "l1": complexity_l1_ds,
    "hscore_full": hs_full,
    "transrate_full": tr_full,
}).reset_index()

plot_df2 = df2.sort_values("n2", ascending=False).reset_index(drop=True)

x = np.arange(len(plot_df2))
w = 0.2

fig, ax = plt.subplots(figsize=(16, 7))

ax.bar(x - 1.5*w, plot_df2["n2"], width=w, label="N2")
ax.bar(x - 0.5*w, plot_df2["l1"], width=w, label="L1")
ax.bar(x + 0.5*w, plot_df2["hscore_full"], width=w, label="H-score")
ax.bar(x + 1.5*w, plot_df2["transrate_full"], width=w, label="TransRate")

ax.set_xticks(x, plot_df2["dataset"], rotation=45, ha="right")
ax.set_ylabel("Mean runtime [s]")
ax.set_title("Dataset-level runtime comparison: selected criteria")
ax.grid(ls=":", c=(.7, .7, .7), axis="y")
ax.spines[['right', 'top']].set_visible(False)
ax.legend(frameon=True)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/runtime_dataset_comparison_selected.png", dpi=220)
plt.savefig(f"{OUT_DIR}/runtime_dataset_comparison_selected.eps")
plt.close()
