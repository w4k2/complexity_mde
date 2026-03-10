import os
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

import problexity as px
from sklearn.model_selection import RepeatedStratifiedKFold

import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from mde import DeepInsight, Norm2Scaler
from utils import Data
from utils import transrate
from utils import hscore

SEED = 1410
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAMES = [
    'australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin'
]

PIXELS = (224, 224)

# repeats for timing
N_REPEATS_COMPLEXITY = 3
N_REPEATS_EXTRACTION = 3
N_REPEATS_ENCODING = 3
N_REPEATS_METRIC = 5


RESULTS_DIR = Path("results_runtime")
FIG_DIR = Path("figures_runtime")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def timed_repeats(fn, *args, repeats=5, warmup=True, **kwargs):
    if warmup:
        _ = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    first_out = None

    for i in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        if i == 0:
            first_out = out

    return first_out, float(np.mean(times)), float(np.std(times)), times


def build_feature_extractor(init_mode: str):
    if init_mode == "imagenet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif init_mode == "random":
        model = resnet18(weights=None)
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    model.eval().to(DEVICE)
    extractor = create_feature_extractor(model, return_nodes={"flatten": "features"})
    extractor.eval().to(DEVICE)
    return extractor


@torch.no_grad()
def extract_features(extractor, X_tensor, batch_size=32):
    feats = []
    for start in range(0, len(X_tensor), batch_size):
        xb = X_tensor[start:start + batch_size].to(DEVICE)
        out = extractor(xb)["features"].detach().cpu().numpy()
        feats.append(out)
    return np.concatenate(feats, axis=0).astype(np.float64)


def encode_deepinsight(X_train, pixels=PIXELS):
    ln = Norm2Scaler()
    di = DeepInsight(
        feature_extractor='pca',
        discretization='bin',
        pixels=pixels
    )

    X_scaled = ln.fit_transform(X_train)
    X_img = di.fit_transform(X_scaled)
    X_img = torch.from_numpy(np.moveaxis(X_img, 3, 1)).float()
    return X_img


def compute_all_complexity_measures(X, y):
    cc = px.ComplexityCalculator()
    cc.fit(X, y)
    report = cc.report()
    metric_names = list(cc._metrics())
    flat = flatten_dict(report)

    metric_values = {}
    for m in metric_names:
        if m in flat:
            metric_values[m] = flat[m]
        else:
            hits = [v for k, v in flat.items() if k.split(".")[-1] == m]
            metric_values[m] = hits[0] if len(hits) > 0 else np.nan
    return metric_values


def dataset_cache_path(dataset_name):
    return RESULTS_DIR / f"features_{dataset_name}.npz"

def load_dataset_cache(path):
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    return {}



def main():
    data = Data(selection=DATASET_NAMES, path="datasets/")
    datasets = data.load()

    extractors = {
        "imagenet": build_feature_extractor("imagenet"),
        "random": build_feature_extractor("random"),
    }

    runtime_rows = []
    complexity_rows = []

    total_steps = len(DATASET_NAMES) * 10 * 4
    pbar = tqdm(total=total_steps, desc="All experiments")

    for dataset_name in DATASET_NAMES:
        X, y = datasets[dataset_name][0], datasets[dataset_name][1]

        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=SEED)
        splits = list(rskf.split(X, y))

        dataset_cache_file = dataset_cache_path(dataset_name)
        dataset_cache = load_dataset_cache(dataset_cache_file)
        dataset_cache_changed = False

        for fold_id, (train_index, test_index) in enumerate(splits):
            pbar.set_postfix(dataset=dataset_name, fold=fold_id)

            X_train = X[train_index]
            y_train = y[train_index].astype(np.int64)

            diff_values, diff_mean, diff_std, _ = timed_repeats(
                compute_all_complexity_measures,
                X_train,
                y_train,
                repeats=N_REPEATS_COMPLEXITY,
                warmup=True,
            )

            runtime_rows.append({
                "dataset": dataset_name,
                "fold": fold_id,
                "method": "complexity_22",
                "init": "raw_tabular",
                "mean_seconds": diff_mean,
                "std_seconds": diff_std,
                "n_samples": int(X_train.shape[0]),
                "n_features": int(X_train.shape[1]),
                "size_proxy": int(X_train.shape[0] * X_train.shape[1]),
            })

            diff_record = {"dataset": dataset_name, "fold": fold_id}
            diff_record.update(diff_values)
            complexity_rows.append(diff_record)
            pbar.update(1)

            X_train_img, enc_mean, enc_std, _ = timed_repeats(
                encode_deepinsight,
                X_train,
                pixels=PIXELS,
                repeats=N_REPEATS_ENCODING,
                warmup=True,
            )

            runtime_rows.append({
                "dataset": dataset_name,
                "fold": fold_id,
                "method": "encoding",
                "init": "shared",
                "mean_seconds": enc_mean,
                "std_seconds": enc_std,
                "n_samples": int(X_train.shape[0]),
                "n_features": int(X_train.shape[1]),
                "size_proxy": int(X_train.shape[0] * X_train.shape[1]),
            })
            pbar.update(1)

            for init_mode in ["imagenet", "random"]:
                # feature extraction timing
                X_feat, feat_mean, feat_std, _ = timed_repeats(
                    extract_features,
                    extractors[init_mode],
                    X_train_img,
                    batch_size=8,
                    repeats=N_REPEATS_EXTRACTION,
                    warmup=True,
                )
                y_feat = y_train.copy()

                runtime_rows.append({
                    "dataset": dataset_name,
                    "fold": fold_id,
                    "method": "feature_extraction",
                    "init": init_mode,
                    "mean_seconds": feat_mean,
                    "std_seconds": feat_std,
                    "n_samples": int(X_feat.shape[0]),
                    "n_features": int(X_feat.shape[1]),
                    "size_proxy": int(X_feat.shape[0] * X_feat.shape[1]),
                })

                # H-score timing
                _, hs_mean, hs_std, _ = timed_repeats(
                    hscore.hscore,
                    X_feat,
                    y_feat,
                    repeats=N_REPEATS_METRIC,
                    warmup=True,
                )
                runtime_rows.append({
                    "dataset": dataset_name,
                    "fold": fold_id,
                    "method": "hscore",
                    "init": init_mode,
                    "mean_seconds": hs_mean,
                    "std_seconds": hs_std,
                    "n_samples": int(X_feat.shape[0]),
                    "n_features": int(X_feat.shape[1]),
                    "size_proxy": int(X_feat.shape[0] * X_feat.shape[1]),
                })

                # TransRate timing
                _, tr_mean, tr_std, _ = timed_repeats(
                    transrate,
                    X_feat,
                    y_feat,
                    repeats=N_REPEATS_METRIC,
                    warmup=True,
                )
                runtime_rows.append({
                    "dataset": dataset_name,
                    "fold": fold_id,
                    "method": "transrate",
                    "init": init_mode,
                    "mean_seconds": tr_mean,
                    "std_seconds": tr_std,
                    "n_samples": int(X_feat.shape[0]),
                    "n_features": int(X_feat.shape[1]),
                    "size_proxy": int(X_feat.shape[0] * X_feat.shape[1]),
                })

                pbar.update(1)

    pbar.close()

    runtime_df = pd.DataFrame(runtime_rows)
    complexity_df = pd.DataFrame(complexity_rows)

    runtime_df.to_csv(RESULTS_DIR / "runtime_summary_per_fold_2.csv", index=False)
    complexity_df.to_csv(RESULTS_DIR / "complexity_values_per_fold_2.csv", index=False)

    global_runtime_df = (
        runtime_df.groupby(["method", "init"])["mean_seconds"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    global_runtime_df.to_csv(RESULTS_DIR / "runtime_global_2.csv", index=False)
    print("\nGlobal runtime summary:")
    print(global_runtime_df)

    matplotlib.rcParams.update({'font.size': 13})

    complexity_vals = runtime_df[
        (runtime_df["method"] == "complexity_22") &
        (runtime_df["init"] == "raw_tabular")
    ]["mean_seconds"].values
    complexity_mean = float(np.mean(complexity_vals))
    complexity_std = float(np.std(complexity_vals))

    encoding_vals = runtime_df[
        (runtime_df["method"] == "encoding") &
        (runtime_df["init"] == "shared")
    ]["mean_seconds"].values
    encoding_mean = float(np.mean(encoding_vals))
    encoding_std = float(np.std(encoding_vals))

    def metric_stats(method, init_mode):
        vals = runtime_df[
            (runtime_df["method"] == method) &
            (runtime_df["init"] == init_mode)
        ]["mean_seconds"].values
        return float(np.mean(vals)), float(np.std(vals))

    hs_img_mean, hs_img_std = metric_stats("hscore", "imagenet")
    hs_rnd_mean, hs_rnd_std = metric_stats("hscore", "random")
    tr_img_mean, tr_img_std = metric_stats("transrate", "imagenet")
    tr_rnd_mean, tr_rnd_std = metric_stats("transrate", "random")

    labels = [
        "complexity\n22",
        "hscore\nimagenet",
        "hscore\nrandom",
        "transrate\nimagenet",
        "transrate\nrandom",
    ]

    x = np.arange(len(labels))
    width = 0.75

    bottom_vals = np.array([
        0.0,
        encoding_mean,
        encoding_mean,
        encoding_mean,
        encoding_mean,
    ])

    top_vals = np.array([
        complexity_mean,
        hs_img_mean,
        hs_rnd_mean,
        tr_img_mean,
        tr_rnd_mean,
    ])

    fig, ax = plt.subplots(figsize=(10, 5))

    # lower stack
    ax.bar(
        x,
        bottom_vals,
        width=width,
        label="encoding",
    )

    # upper stack
    ax.bar(
        x,
        top_vals,
        width=width,
        bottom=bottom_vals,
        label="metric",
    )

    # annotate totals
    totals = bottom_vals + top_vals
    for xi, total in zip(x, totals):
        ax.text(xi, total * 1.05, f"{total:.3g}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean runtime [s]")
    ax.set_title("Runtime comparison: encoding + metric (feature extraction excluded)")
    ax.grid(ls=":", c=(.7, .7, .7), axis="y")
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_yscale("log")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "runtime_stacked_encoding_metric.png", dpi=200)
    plt.savefig(FIG_DIR / "runtime_stacked_encoding_metric.eps")
    plt.close()


    total_means = totals

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, total_means, width=width)

    for xi, total in zip(x, total_means):
        ax.text(xi, total * 1.05, f"{total:.3g}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean runtime [s]")
    ax.set_title("Total runtime comparison (encoding + metric)")
    ax.grid(ls=":", c=(.7, .7, .7), axis="y")
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "runtime_total_bars.png", dpi=200)
    plt.savefig(FIG_DIR / "runtime_total_bars.eps")
    plt.close()

    box_labels = [
        "complexity\n22",
        "encoding",
        "hscore\nimagenet",
        "hscore\nrandom",
        "transrate\nimagenet",
        "transrate\nrandom",
    ]

    box_data = [
        runtime_df[(runtime_df["method"] == "complexity_22") & (runtime_df["init"] == "raw_tabular")]["mean_seconds"].values,
        runtime_df[(runtime_df["method"] == "encoding") & (runtime_df["init"] == "shared")]["mean_seconds"].values,
        runtime_df[(runtime_df["method"] == "hscore") & (runtime_df["init"] == "imagenet")]["mean_seconds"].values,
        runtime_df[(runtime_df["method"] == "hscore") & (runtime_df["init"] == "random")]["mean_seconds"].values,
        runtime_df[(runtime_df["method"] == "transrate") & (runtime_df["init"] == "imagenet")]["mean_seconds"].values,
        runtime_df[(runtime_df["method"] == "transrate") & (runtime_df["init"] == "random")]["mean_seconds"].values,
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(box_data, tick_labels=box_labels, showfliers=False)
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtime distribution across datasets and folds")
    ax.grid(ls=":", c=(.7, .7, .7), axis="y")
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "runtime_boxplot.png", dpi=200)
    plt.savefig(FIG_DIR / "runtime_boxplot.eps")
    plt.close()

if __name__ == "__main__":
    main()