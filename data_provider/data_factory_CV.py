from data_provider.data_loader_CV import PPMI_Dataset,Mātai_Dataset,Neurocon_Dataset,Taowu_Dataset,Abide_Dataset,ADNI_Dataset,MDD_Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from functools import partial
import torch
import numpy as np

data_dict = {
    'PPMI': PPMI_Dataset,
    'Mātai': Mātai_Dataset,
    'Neurocon': Neurocon_Dataset,
    'Taowu': Taowu_Dataset,
    'Abide': Abide_Dataset,
    'ADNI': ADNI_Dataset,
    'MDD': MDD_Dataset,
}

def _fit_time_length(x, max_len, random_crop=False):
    """按训练入口设置的 seq_len 对齐时间维：长序列截断/随机裁剪，短序列末尾补 0。"""
    if x.shape[0] >= max_len:
        if random_crop and x.shape[0] > max_len:
            start = torch.randint(0, x.shape[0] - max_len + 1, (1,)).item()
            return x[start : start + max_len]
        return x[:max_len]
    raise ValueError(
        f"Time series length {x.shape[0]} is shorter than seq_len={max_len}. "
        "Short samples should be filtered before DataLoader collation."
    )


def collate_fn(batch, max_len, random_crop=False):
    field_count = len(batch[0])
    if field_count == 4:
        data, labels, correlations, sites = zip(*batch)
    elif field_count == 3:
        third = batch[0][2]
        if getattr(third, "ndim", 0) == 2:
            data, labels, correlations = zip(*batch)
            sites = None
        else:
            data, labels, sites = zip(*batch)
            correlations = None
    else:
        data, labels = zip(*batch)
        correlations = None
        sites = None
    padded_data = [_fit_time_length(x, max_len, random_crop=random_crop) for x in data]
    result = [torch.stack(padded_data), torch.tensor(labels)]
    if correlations is not None:
        result.append(torch.stack(correlations))
    if sites is not None:
        result.append(torch.tensor(sites))
    return tuple(result)

def custom_collate_fn(batch, max_len, random_crop=False):
    return collate_fn(batch, max_len=max_len, random_crop=random_crop)


class HarmonizedSubset(torch.utils.data.Dataset):
    """按 fold 应用输入时序 harmonization，统计量只来自训练集。"""

    def __init__(self, dataset, indices, site_stats=None, global_stats=None, site_to_index=None, return_site=False):
        self.dataset = dataset
        self.indices = list(indices)
        self.site_stats = site_stats or {}
        self.global_stats = global_stats
        self.site_to_index = site_to_index or {}
        self.return_site = bool(return_site)

    def __len__(self):
        return len(self.indices)

    def _harmonize_signal(self, signal, site_id):
        if self.global_stats is None:
            return signal
        mean, std = self.site_stats.get(site_id, self.global_stats)
        mean = mean.to(device=signal.device, dtype=signal.dtype)
        std = std.to(device=signal.device, dtype=signal.dtype)
        return (signal - mean) / std.clamp_min(1e-6)

    def __getitem__(self, item):
        dataset_idx = int(self.indices[item])
        sample = self.dataset[dataset_idx]
        site_id = getattr(self.dataset, "site_ids", ["unknown"] * len(self.dataset))[dataset_idx]
        site_label = torch.tensor(self.site_to_index.get(site_id, 0), dtype=torch.long)
        if len(sample) == 3:
            signal, label, correlation = sample
            result = (self._harmonize_signal(signal, site_id), label, correlation)
        else:
            signal, label = sample
            result = (self._harmonize_signal(signal, site_id), label)
        if self.return_site:
            return (*result, site_label)
        return result


class SiteLabelSubset(torch.utils.data.Dataset):
    """不做时序校正，仅在需要站点对抗训练时追加 site label。"""

    def __init__(self, dataset, indices, site_to_index):
        self.dataset = dataset
        self.indices = list(indices)
        self.site_to_index = site_to_index

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        dataset_idx = int(self.indices[item])
        sample = self.dataset[dataset_idx]
        site_id = getattr(self.dataset, "site_ids", ["unknown"] * len(self.dataset))[dataset_idx]
        site_label = torch.tensor(self.site_to_index.get(site_id, 0), dtype=torch.long)
        return (*sample, site_label)


def _fit_site_zscore_stats(dataset, train_idx, min_samples_per_site=2):
    """只用训练 fold 估计每个站点、每个 ROI 的输入时序均值和标准差。"""

    site_ids = getattr(dataset, "site_ids", None)
    if site_ids is None:
        return {}, None

    grouped = {}
    all_signals = []
    for idx in train_idx:
        signal = dataset.data[int(idx)]
        site_id = site_ids[int(idx)]
        grouped.setdefault(site_id, []).append(signal)
        all_signals.append(signal)

    if not all_signals:
        return {}, None

    # 不同数据集/被试的时间长度可能不同；站点校正只需要每个 ROI
    # 在训练 fold 内的总体均值和方差，因此沿时间维拼接比 stack 更稳。
    global_cat = torch.cat(all_signals, dim=0)
    global_mean = global_cat.mean(dim=0, keepdim=True)
    global_std = global_cat.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    global_stats = (global_mean, global_std)

    site_stats = {}
    for site_id, signals in grouped.items():
        if len(signals) < int(min_samples_per_site):
            continue
        cat = torch.cat(signals, dim=0)
        mean = cat.mean(dim=0, keepdim=True)
        std = cat.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        site_stats[site_id] = (mean, std)
    return site_stats, global_stats


def _format_site_label_distribution(dataset, indices, unique_sites, unique_labels):
    """统计一个 fold split 内的站点 × 标签数量，用于检查 site-label 混杂。"""

    site_ids = getattr(dataset, "site_ids", ["unknown"] * len(dataset))
    labels = getattr(dataset, "labels", [])
    counts = {
        site_id: {int(label): 0 for label in unique_labels}
        for site_id in unique_sites
    }
    for idx in indices:
        idx = int(idx)
        site_id = site_ids[idx] if idx < len(site_ids) else "unknown"
        label = int(labels[idx])
        counts.setdefault(site_id, {int(item): 0 for item in unique_labels})
        counts[site_id][label] = counts[site_id].get(label, 0) + 1

    lines = []
    header = "site".ljust(10) + " | " + " | ".join(f"class {int(label)}".rjust(8) for label in unique_labels)
    lines.append(header)
    lines.append("-" * len(header))
    for site_id in sorted(counts):
        values = " | ".join(str(counts[site_id].get(int(label), 0)).rjust(8) for label in unique_labels)
        lines.append(site_id.ljust(10) + " | " + values)
    return "\n".join(lines)


def data_provider(args):
    Data = data_dict[args.data]
    kf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=int(getattr(args, "seed", 2024)))
    
    module2_disabled = int(getattr(args, "use_causal_module2", 1)) == 0
    hyperbolic_enabled = int(
        getattr(args, "use_hyperbolic_modules34", getattr(args, "use_hgcn_module3", 0))
    ) == 1
    gcn_fallback_enabled = not hyperbolic_enabled
    graph_path_needs_adjacency = hyperbolic_enabled or gcn_fallback_enabled
    correlation_fallback_enabled = bool(
        getattr(args, "use_sample_correlation_when_module2_disabled", 1)
    )
    module2_correlation_blend = float(getattr(args, "module2_sample_correlation_blend", 0.0) or 0.0)
    fc_readout_branch_enabled = bool(getattr(args, "use_fc_readout_branch", 0))
    use_sample_correlation = (
        module2_disabled and graph_path_needs_adjacency and correlation_fallback_enabled
    ) or (
        (not module2_disabled) and graph_path_needs_adjacency and module2_correlation_blend > 0.0
    ) or fc_readout_branch_enabled
    dataset = Data(
        args.data_path,
        args.data_type,
        args.protocol,
        args.seq_len,
        use_sample_correlation=use_sample_correlation,
    )
    labels = np.asarray([int(label) for label in dataset.labels])
    site_ids = getattr(dataset, "site_ids", ["unknown"] * len(dataset))
    unique_sites = sorted(set(site_ids))
    args.site_count = max(len(unique_sites), 1)
    site_to_index = {site_id: idx for idx, site_id in enumerate(unique_sites)}
    drop_last=False
    
    unique_labels = np.unique(np.array(dataset.labels))
    num_categories = len(unique_labels)
    
    train_loaders=[]
    val_loaders=[]
    
    return_site = bool(
        getattr(args, "use_site_adversarial", 0) or getattr(args, "use_site_modulation", 0)
    )
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset.data, labels)):
        harmonization = str(getattr(args, "time_series_harmonization", "none")).lower()
        if harmonization == "site_zscore":
            site_stats, global_stats = _fit_site_zscore_stats(
                dataset,
                train_idx,
                min_samples_per_site=int(getattr(args, "site_harmonization_min_samples", 2)),
            )
            train_data = HarmonizedSubset(
                dataset,
                train_idx,
                site_stats=site_stats,
                global_stats=global_stats,
                site_to_index=site_to_index,
                return_site=return_site,
            )
            val_data = HarmonizedSubset(
                dataset,
                val_idx,
                site_stats=site_stats,
                global_stats=global_stats,
                site_to_index=site_to_index,
                return_site=return_site,
            )
        elif harmonization in ("none", ""):
            if return_site:
                train_data = SiteLabelSubset(dataset, train_idx, site_to_index=site_to_index)
                val_data = SiteLabelSubset(dataset, val_idx, site_to_index=site_to_index)
            else:
                train_data = torch.utils.data.Subset(dataset, train_idx)
                val_data = torch.utils.data.Subset(dataset, val_idx)
        else:
            raise ValueError(
                f"Unsupported time_series_harmonization={harmonization!r}. "
                "Use 'none' or 'site_zscore'."
            )

        train_random_crop = bool(getattr(args, "module1_random_crop", 0))
        num_workers = max(int(getattr(args, "num_workers", 0)), 0)
        pin_memory = bool(getattr(args, "pin_memory", 0)) and bool(getattr(args, "use_gpu", 0))
        persistent_workers = bool(getattr(args, "persistent_workers", 0)) and num_workers > 0
        loader_kwargs = {
            "batch_size": args.batch_size,
            "drop_last": drop_last,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = max(int(getattr(args, "prefetch_factor", 2)), 1)

        train_loader = DataLoader(
            train_data,
            shuffle=True,
            collate_fn=partial(custom_collate_fn, max_len=args.seq_len, random_crop=train_random_crop),
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_data,
            shuffle=False,
            collate_fn=partial(custom_collate_fn, max_len=args.seq_len, random_crop=False),
            **loader_kwargs,
        )
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

        if args.print_data_info:
            
            train_labels = [train_data[i][1] for i in range(len(train_data))]
            train_samples_num=[]
            for i in range(num_categories):
                train_samples_num.append(train_labels.count(i))

            val_labels = [val_data[i][1] for i in range(len(val_data))]
            val_samples_num=[]
            for i in range(num_categories):
                val_samples_num.append(val_labels.count(i))

            print(f"Fold {fold + 1}:")
            print(f"  Training samples: {len(train_data)}")
            for i in range(num_categories):
                print(f'Number of Class {i} in training set: {train_samples_num[i]}')
            print(f"  Validation samples: {len(val_data)}")
            for i in range(num_categories):
                print(f'Number of Class {i} in validation set: {val_samples_num[i]}')
            print("  Training site x label distribution:")
            print(_format_site_label_distribution(dataset, train_idx, unique_sites, unique_labels))
            print("  Validation site x label distribution:")
            print(_format_site_label_distribution(dataset, val_idx, unique_sites, unique_labels))
            
            sample_batch = next(iter(train_loader))
            sample_data, sample_label = sample_batch[:2]
            print(f"Sample data shape: {sample_data.shape}, Sample label: {sample_label}")
            if len(sample_batch) == 3:
                print(f"Sample correlation matrix shape: {sample_batch[2].shape}")
    return train_loaders,val_loaders
