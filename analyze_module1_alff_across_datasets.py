import argparse
import math
from pathlib import Path

import numpy as np
import scipy.io


def load_timeseries(mat_path):
    mat = scipy.io.loadmat(mat_path)
    for key in ("data", "x", "ts", "timeseries", "time_series"):
        if key in mat:
            arr = np.asarray(mat[key], dtype=np.float32)
            if arr.ndim == 2:
                return arr
            if arr.ndim == 3 and arr.shape[0] == 1:
                return arr[0]
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if len(keys) == 1:
        arr = np.asarray(mat[keys[0]], dtype=np.float32)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
    raise KeyError(f"Cannot infer time series from {mat_path}")


def normalize_orientation(ts, channel_hint=116):
    arr = np.asarray(ts, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    if arr.shape[0] == channel_hint:
        return arr.T
    if arr.shape[1] == channel_hint:
        return arr
    return arr if arr.shape[0] >= arr.shape[1] else arr.T


def alff_features(x, tr=2.0, low_hz=0.01, high_hz=0.08):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    spec = np.fft.rfft(x, axis=0)
    amp = np.abs(spec)
    power = amp ** 2
    freqs = np.fft.rfftfreq(n, d=max(tr, 1e-6))
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        band_mask = freqs > 0
    if not np.any(band_mask):
        band_mask = np.ones_like(freqs, dtype=bool)
    non_dc = freqs > 0
    band_amp = amp[band_mask]
    band_power = power[band_mask]
    total_power = power[non_dc] if np.any(non_dc) else power
    alff = band_amp.mean(axis=0)
    falff = band_power.sum(axis=0) / np.clip(total_power.sum(axis=0), 1e-6, None)
    band_std = band_amp.std(axis=0)
    dom_idx = band_amp.argmax(axis=0)
    dom_freq = freqs[band_mask][dom_idx]
    temp_std = x.std(axis=0)
    return np.stack(
        [
            np.log1p(alff),
            falff,
            np.log1p(band_std),
            dom_freq,
            temp_std,
        ],
        axis=-1,
    )


def simple_signal_stats(x):
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x, axis=0)
    return np.stack(
        [
            x.mean(axis=0),
            x.std(axis=0),
            np.abs(dx).mean(axis=0),
            np.std(dx, axis=0),
            x.min(axis=0),
            x.max(axis=0),
        ],
        axis=-1,
    )


def flatten_dataset(paths, sample_limit=None, channel_hint=116):
    items = []
    for p in paths:
        ts = normalize_orientation(load_timeseries(p), channel_hint=channel_hint)
        items.append(ts)
        if sample_limit is not None and len(items) >= sample_limit:
            break
    return items


def collect_dataset(root, protocol="AAL116", sample_limit=200):
    root = Path(root)
    paths = sorted(root.rglob(f"*_{protocol}_features_timeseries.mat"))
    if not paths:
        return []
    return flatten_dataset(paths[:sample_limit], sample_limit=sample_limit)


def summarize_embeddings(embeddings):
    flat = embeddings.reshape(embeddings.shape[0], -1)
    return {
        "mean": flat.mean(axis=0),
        "cov": np.cov(flat, rowvar=False),
    }


def mean_cov_distance(a, b):
    mean_dist = float(np.linalg.norm(a["mean"] - b["mean"]))
    cov_a = np.asarray(a["cov"], dtype=np.float64)
    cov_b = np.asarray(b["cov"], dtype=np.float64)
    cov_dist = float(np.linalg.norm(cov_a - cov_b, ord="fro"))
    return mean_dist, cov_dist


def pairwise_matrix(dataset_stats):
    names = list(dataset_stats.keys())
    rows = []
    for i, name_a in enumerate(names):
        row = []
        for j, name_b in enumerate(names):
            if i == j:
                row.append((0.0, 0.0))
            else:
                row.append(mean_cov_distance(dataset_stats[name_a], dataset_stats[name_b]))
        rows.append(row)
    return names, rows


def print_matrix(title, names, matrix):
    print(f"\n{title}")
    header = "dataset".ljust(14) + "".join(name[:10].rjust(24) for name in names)
    print(header)
    for name, row in zip(names, matrix):
        line = name.ljust(14)
        for mean_dist, cov_dist in row:
            line += f"{mean_dist:8.3f}/{cov_dist:8.3f}".rjust(24)
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Compare raw signals vs ALFF/fALFF across datasets.")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--protocol", default="AAL116")
    parser.add_argument("--sample-limit", type=int, default=120)
    parser.add_argument("--tr", type=float, default=2.0)
    parser.add_argument("--low-hz", type=float, default=0.01)
    parser.add_argument("--high-hz", type=float, default=0.08)
    args = parser.parse_args()

    dataset_roots = [
        "Abide",
        "MDD",
        "Mātai",
        "Neurocon",
        "PPMI",
        "Taowu",
    ]
    raw_stats = {}
    alff_stats = {}
    for dataset_name in dataset_roots:
        ds_root = Path(args.data_root) / dataset_name
        if not ds_root.exists():
            continue
        samples = collect_dataset(ds_root, protocol=args.protocol, sample_limit=args.sample_limit)
        if not samples:
            continue
        raw_desc = np.stack([simple_signal_stats(s) for s in samples], axis=0)
        alff_desc = np.stack([alff_features(s, tr=args.tr, low_hz=args.low_hz, high_hz=args.high_hz) for s in samples], axis=0)
        raw_stats[dataset_name] = summarize_embeddings(raw_desc)
        alff_stats[dataset_name] = summarize_embeddings(alff_desc)
        print(f"{dataset_name}: n={len(samples)}")

    if len(raw_stats) < 2:
        print("Not enough datasets for comparison.")
        return

    names, raw_matrix = pairwise_matrix(raw_stats)
    _, alff_matrix = pairwise_matrix(alff_stats)
    print_matrix("Raw signal feature distance (mean/cov)", names, raw_matrix)
    print_matrix("ALFF/fALFF feature distance (mean/cov)", names, alff_matrix)

    raw_mean = np.mean([m for row in raw_matrix for m in row if m != (0.0, 0.0)], axis=0)
    alff_mean = np.mean([m for row in alff_matrix for m in row if m != (0.0, 0.0)], axis=0)
    print(
        f"\nAverage pairwise distance raw: mean={raw_mean[0]:.3f}, cov={raw_mean[1]:.3f}"
    )
    print(
        f"Average pairwise distance alff: mean={alff_mean[0]:.3f}, cov={alff_mean[1]:.3f}"
    )
    ratio_mean = alff_mean[0] / max(raw_mean[0], 1e-6)
    ratio_cov = alff_mean[1] / max(raw_mean[1], 1e-6)
    print(f"ALFF/raw ratio: mean={ratio_mean:.3f}, cov={ratio_cov:.3f}")


if __name__ == "__main__":
    main()
