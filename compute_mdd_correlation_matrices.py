"""Compute Pearson correlation matrices for MDD fMRI time-series files.

The paper constructs functional connectivity (FC) by computing Pearson
correlation coefficients between ROI BOLD signals. MDD files store time series
as [time_points, roi_count], so each output matrix is [roi_count, roi_count].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


TIME_SERIES_SUFFIX = "features_timeseries"
CORRELATION_SUFFIX = "correlation_matrix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute sample-level Pearson correlation matrices for dataset/MDD."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/MDD"),
        help="MDD dataset root directory.",
    )
    parser.add_argument(
        "--protocol",
        default="AAL116",
        help='Atlas/protocol name, for example "AAL116"; use "all" for every protocol.',
    )
    parser.add_argument(
        "--data-key",
        default="data",
        help="Variable name used for reading time series and saving correlation matrices.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing correlation_matrix .mat files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list how many files would be processed.",
    )
    return parser.parse_args()


def load_time_series(path: Path, data_key: str) -> np.ndarray:
    mat_data = loadmat(path)
    if data_key in mat_data:
        data = mat_data[data_key]
    else:
        keys = [key for key in mat_data if not key.startswith("__")]
        if len(keys) != 1:
            raise KeyError(
                f"Cannot find {data_key!r} in {path}. Available matrix keys: {keys}"
            )
        data = mat_data[keys[0]]

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D time-series matrix in {path}, got {data.shape}.")
    return data


def pearson_correlation_matrix(time_series: np.ndarray) -> np.ndarray:
    """Return ROI-wise Pearson correlation for data shaped [T, N]."""
    centered = time_series - np.mean(time_series, axis=0, keepdims=True)
    covariance = centered.T @ centered
    sum_squares = np.sum(centered * centered, axis=0)
    denominator = np.sqrt(np.outer(sum_squares, sum_squares))

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = covariance / denominator

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    non_constant = sum_squares > 0
    corr[np.diag_indices_from(corr)] = non_constant.astype(np.float64)
    return corr


def output_path_for(time_series_path: Path) -> Path:
    name = time_series_path.name
    expected = f"_{TIME_SERIES_SUFFIX}.mat"
    if not name.endswith(expected):
        raise ValueError(f"Unexpected time-series filename: {time_series_path}")
    return time_series_path.with_name(name[: -len(expected)] + f"_{CORRELATION_SUFFIX}.mat")


def iter_time_series_files(root: Path, protocol: str) -> list[Path]:
    if protocol.lower() == "all":
        pattern = f"*_{TIME_SERIES_SUFFIX}.mat"
    else:
        pattern = f"*_{protocol}_{TIME_SERIES_SUFFIX}.mat"
    return sorted(root.rglob(pattern))


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"MDD dataset root does not exist: {root}")

    files = iter_time_series_files(root, args.protocol)
    print(f"Found {len(files)} time-series files under {root}")
    if args.dry_run:
        for path in files[:10]:
            print(f"  {path} -> {output_path_for(path)}")
        if len(files) > 10:
            print(f"  ... {len(files) - 10} more")
        return 0

    saved = 0
    skipped = 0
    failed: list[tuple[Path, Exception]] = []

    for path in files:
        out_path = output_path_for(path)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            time_series = load_time_series(path, args.data_key)
            corr = pearson_correlation_matrix(time_series)
            savemat(out_path, {args.data_key: corr})
            saved += 1
        except Exception as exc:  # keep processing independent samples
            failed.append((path, exc))

    print(f"Saved: {saved}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {len(failed)}")
    for path, exc in failed[:20]:
        print(f"  {path}: {exc}")
    if len(failed) > 20:
        print(f"  ... {len(failed) - 20} more failures")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
