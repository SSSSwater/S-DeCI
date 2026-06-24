import os
import re
import scipy.io
import torch
import numpy as np
from torch.utils.data import Dataset

TIME_SERIES_SUFFIX = "features_timeseries"
CORRELATION_SUFFIX = "correlation_matrix"


def _load_mat_tensor(file_path, role="data"):
    mat_data = scipy.io.loadmat(file_path)
    keys = [key for key in mat_data.keys() if not key.startswith("__")]
    key_candidates = ["data", role, CORRELATION_SUFFIX, "corr", "correlation", "matrix"]
    for key in key_candidates:
        if key in mat_data:
            return torch.tensor(mat_data[key], dtype=torch.float32)
    if len(keys) == 1:
        return torch.tensor(mat_data[keys[0]], dtype=torch.float32)
    raise KeyError(
        f"Cannot find matrix data in {file_path}. "
        f"Expected one of {key_candidates}, available keys: {keys}."
    )


def _subject_prefix_from_file(file_name, protocol):
    protocol_marker = f"_{protocol}_"
    if protocol_marker in file_name:
        return file_name.split(protocol_marker, 1)[0]
    suffix_marker = f"_{TIME_SERIES_SUFFIX}"
    if suffix_marker in file_name:
        return file_name.split(suffix_marker, 1)[0]
    return os.path.splitext(file_name)[0]


def _find_correlation_matrix_file(sample_file_path, protocol):
    sample_dir = os.path.dirname(sample_file_path)
    sample_name = os.path.basename(sample_file_path)
    subject_prefix = _subject_prefix_from_file(sample_name, protocol)
    tried_patterns = [
        f"{subject_prefix}_{protocol}_{CORRELATION_SUFFIX}.mat",
        f"{subject_prefix}_{CORRELATION_SUFFIX}.mat",
        f"{subject_prefix}_features_sub_{CORRELATION_SUFFIX}.mat",
    ]

    for candidate_name in tried_patterns:
        candidate_path = os.path.join(sample_dir, candidate_name)
        if os.path.exists(candidate_path):
            return candidate_path

    candidates = []
    for file_name in os.listdir(sample_dir):
        is_mat = file_name.endswith(".mat")
        has_subject = file_name.startswith(subject_prefix)
        has_correlation = CORRELATION_SUFFIX in file_name or "features_sub_correlation_matrix" in file_name
        if is_mat and has_subject and has_correlation:
            candidates.append(os.path.join(sample_dir, file_name))
    if candidates:
        protocol_matches = [path for path in candidates if f"_{protocol}_" in os.path.basename(path)]
        return sorted(protocol_matches or candidates)[0]

    raise FileNotFoundError(
        f"Correlation matrix not found for sample: {sample_file_path}. "
        f"Tried patterns: {tried_patterns}."
    )


def _site_id_from_file_path(file_path):
    """从多站点 fMRI 文件名中解析站点 id。

    MDD 文件通常形如 `sub-control_s17_1_0028_AAL116_features_timeseries.mat`，
    其中 `s17` 对应采集站点。其他数据集没有明确站点时回退为 `unknown`。
    """

    file_name = os.path.basename(file_path).lower()
    match = re.search(r"_(s\d+)[_-]", file_name)
    if match:
        return match.group(1)
    match = re.search(r"(site\d+|site[-_][a-z0-9]+)", file_name)
    if match:
        return match.group(1).replace("-", "_")
    return "unknown"


class CorrelationFallbackMixin:
    def _init_storage(self, use_sample_correlation=False, seq_len=None, data_type="TS"):
        self.data = []
        self.labels = []
        self.sample_paths = []
        self.site_ids = []
        self.correlation_matrices = []
        self.use_sample_correlation = bool(use_sample_correlation)
        self.seq_len = None if seq_len is None else int(seq_len)
        self.data_type = str(data_type).upper()
        self.skipped_short_count = 0

    def _append_sample(self, file_path, label, signal):
        if self.data_type == "TS" and self.seq_len is not None and signal.shape[0] < self.seq_len:
            self.skipped_short_count += 1
            return False
        self.data.append(signal)
        self.labels.append(torch.tensor(label, dtype=torch.long))
        self.sample_paths.append(file_path)
        self.site_ids.append(_site_id_from_file_path(file_path))
        if not self.use_sample_correlation:
            return True
        correlation_path = _find_correlation_matrix_file(file_path, self.protocol)
        correlation = _load_mat_tensor(correlation_path, role=CORRELATION_SUFFIX)
        expected_shape = (signal.shape[-1], signal.shape[-1])
        if tuple(correlation.shape) != expected_shape:
            raise ValueError(
                f"Correlation matrix shape mismatch for sample {file_path}: "
                f"expected {expected_shape}, got {tuple(correlation.shape)} from {correlation_path}."
            )
        self.correlation_matrices.append(correlation)
        return True

    def _get_sample(self, idx):
        if self.use_sample_correlation:
            return self.data[idx], self.labels[idx], self.correlation_matrices[idx]
        return self.data[idx], self.labels[idx]

class PPMI_Dataset(Dataset):
    """
    A custom PyTorch dataset for loading time series data from PPMI (Parkinson's Progression Markers Initiative) dataset.
    
    Args:
        data_type: Choose from [TS: raw time series, FC: functional connectivity]
        protocol: ROI (region of interest) number, 'schaefer100' for 100, 'AAL116' for 116, 'harvard48' for 48, 'ward100' for 100, 'kmeans100' for 100
        Length of this dataset is T=210
        Total 182 samples. (Importantly, we removed some outlier with length != 210 )
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/ppmi", data_type='TS',protocol="schaefer100",seq_len=210, use_sample_correlation=False):
        self.source_dir = source_dir
        self.protocol = protocol
        # 0: control 1:patient 2: prodromal 3:swedd
        self.categories = ['control', 'patient', 'prodromal', 'swedd']
        CorrelationFallbackMixin._init_storage(self, use_sample_correlation, seq_len=seq_len, data_type=data_type)
        file_name_dict={'TS':"features_timeseries",
                        'FC':"correlation_matrix"}
        self.filename=str(protocol)+'_'+str(file_name_dict[data_type])+'.mat'
        self.load_data()

    def load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            subfolders = os.listdir(category_dir)
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_dir, subfolder)
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(self.filename):
                        file_path = os.path.join(subfolder_path, file_name)
                        signal = _load_mat_tensor(file_path)
                        CorrelationFallbackMixin._append_sample(self, file_path, label, signal)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return CorrelationFallbackMixin._get_sample(self, idx)
    
    
class Mātai_Dataset(Dataset):
    """
    A custom dataset class for loading time-series data from the Mātai dataset.
    
    Args:
        data_type: Choose from [TS: raw time series, FC: functional connectivity]
        protocol: ROI (region of interest) number, 'schaefer100' for 100, 'AAL116' for 116, 'harvard48' for 48, 'ward100' for 100, 'kmeans100' for 100
        Length of this dataset is T=200
        Total 60 samples.
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/Mātai_dataset", data_type='TS',protocol="schaefer100",seq_len=200, use_sample_correlation=False):
        self.source_dir = source_dir
        self.protocol = protocol
        # 0:baseline 1:postseason
        self.categories = ['baseline', 'postseason']
        CorrelationFallbackMixin._init_storage(self, use_sample_correlation, seq_len=seq_len, data_type=data_type)
        file_name_dict={'TS':"features_timeseries",
                        'FC':"correlation_matrix"}
        self.filename=str(protocol)+'_'+str(file_name_dict[data_type])+'.mat'
        self.load_data()

    def load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            subfolders = os.listdir(category_dir)
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_dir, subfolder)
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(self.filename):
                        file_path = os.path.join(subfolder_path, file_name)
                        signal = _load_mat_tensor(file_path)
                        CorrelationFallbackMixin._append_sample(self, file_path, label, signal)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return CorrelationFallbackMixin._get_sample(self, idx)
    
    
class Neurocon_Dataset(Dataset):
    """
    A custom dataset class for loading time-series data from the Neurocon dataset.
    
    Args:
        data_type: Choose from [TS: raw time series, FC: functional connectivity]
        protocol: ROI (region of interest) number, 'schaefer100' for 100, 'AAL116' for 116, 'harvard48' for 48, 'ward100' for 100, 'kmeans100' for 100
        Length of this dataset is T=137
        Total 41 samples.
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/neurocon", data_type='TS',protocol="schaefer100",seq_len=137, use_sample_correlation=False):
        self.source_dir = source_dir
        self.protocol = protocol
        # 0:control 1:patient
        self.categories = ['control', 'patient']
        CorrelationFallbackMixin._init_storage(self, use_sample_correlation, seq_len=seq_len, data_type=data_type)
        file_name_dict={'TS':"features_timeseries",
                        'FC':"correlation_matrix"}
        self.filename=str(protocol)+'_'+str(file_name_dict[data_type])+'.mat'
        self.load_data()

    def load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            subfolders = os.listdir(category_dir)
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_dir, subfolder)
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(self.filename):
                        file_path = os.path.join(subfolder_path, file_name)
                        signal = _load_mat_tensor(file_path)
                        CorrelationFallbackMixin._append_sample(self, file_path, label, signal)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return CorrelationFallbackMixin._get_sample(self, idx)
    
    
class Taowu_Dataset(Dataset):
    """
    A custom dataset class for loading time-series data from the Taowu dataset.
    
    Args:
        data_type: Choose from [TS: raw time series, FC: functional connectivity]
        protocol: ROI (region of interest) number, 'schaefer100' for 100, 'AAL116' for 116, 'harvard48' for 48, 'ward100' for 100, 'kmeans100' for 100
        Length of this dataset is T=239
        Total 40 samples.
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/taowu", data_type='TS',protocol="schaefer100",seq_len=239, use_sample_correlation=False):
        self.source_dir = source_dir
        self.protocol = protocol
        # 0:control 1:patient
        self.categories = ['control', 'patient']
        CorrelationFallbackMixin._init_storage(self, use_sample_correlation, seq_len=seq_len, data_type=data_type)
        file_name_dict={'TS':"features_timeseries",
                        'FC':"correlation_matrix"}
        self.filename=str(protocol)+'_'+str(file_name_dict[data_type])+'.mat'
        self.load_data()

    def load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            subfolders = os.listdir(category_dir)
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_dir, subfolder)
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(self.filename):
                        file_path = os.path.join(subfolder_path, file_name)
                        signal = _load_mat_tensor(file_path)
                        CorrelationFallbackMixin._append_sample(self, file_path, label, signal)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return CorrelationFallbackMixin._get_sample(self, idx)


class MDD_Dataset(Dataset):
    """
    A custom dataset class for loading time-series data from the MDD dataset.

    Args:
        data_type: Choose from [TS: raw time series, FC: functional connectivity]
        protocol: ROI (region of interest) number, for example AAL116.
        Length of this dataset is T=230 for the current AAL116 files.
    """
    def __init__(self, source_dir="dataset/MDD", data_type='TS', protocol="AAL116", seq_len=230, use_sample_correlation=False):
        self.source_dir = source_dir
        self.protocol = protocol
        # 0:control 1:patient
        self.categories = ['control', 'patient']
        CorrelationFallbackMixin._init_storage(self, use_sample_correlation, seq_len=seq_len, data_type=data_type)
        file_name_dict={'TS':"features_timeseries",
                        'FC':"correlation_matrix"}
        self.filename=str(protocol)+'_'+str(file_name_dict[data_type])+'.mat'
        self.load_data()

    def load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            subfolders = os.listdir(category_dir)
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_dir, subfolder)
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(self.filename):
                        file_path = os.path.join(subfolder_path, file_name)
                        signal = _load_mat_tensor(file_path)
                        CorrelationFallbackMixin._append_sample(self, file_path, label, signal)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return CorrelationFallbackMixin._get_sample(self, idx)

class Abide_Dataset(Dataset):
    """
    A custom dataset class for loading time-series data from the ABIDE dataset.
    
    Args:
        data_type: Choose from [TS: raw time series, FC: functional connectivity]
        protocol: ROI (region of interest) number, 'schaefer100' for 100, 'AAL116' for 116, 'harvard48' for 48, 'ward100' for 100, 'kmeans100' for 100
        Length of this dataset is T=120-300 (Importantly, the original length of the ABIDE time series range from 120 to 300. 
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/abide", data_type='TS',protocol="ward100",seq_len=300, use_sample_correlation=False):
        self.source_dir = source_dir
        self.protocol = protocol
        # 0:control 1:patient
        self.categories = ['control', 'patient']
        CorrelationFallbackMixin._init_storage(self, use_sample_correlation, seq_len=seq_len, data_type=data_type)
        file_name_dict={'TS':"features_timeseries",
                        'FC':"correlation_matrix"}
        self.filename=str(protocol)+'_'+str(file_name_dict[data_type])+'.mat'
        self.load_data()
    def load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            subfolders = os.listdir(category_dir)
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_dir, subfolder)
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(self.filename):
                        file_path = os.path.join(subfolder_path, file_name)
                        signal = _load_mat_tensor(file_path)
                        CorrelationFallbackMixin._append_sample(self, file_path, label, signal)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return CorrelationFallbackMixin._get_sample(self, idx)



class ADNI_Dataset(Dataset):
    """
    A custom dataset class for loading time-series data from the ADNI dataset.
    """
    def __init__(self, source_dir="/data/gqyu/FMRI/dataset/ADNI/ADNI", data_type='TS', protocol="AAL116", seq_len=197):
        self.source_dir = source_dir
        self.categories = ['Control', 'MCI', 'AD']   # 0,1,2
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.source_dir, category)
            for fname in os.listdir(category_dir):
                fpath = os.path.join(category_dir, fname)
                arr = np.load(fpath)
                x = torch.from_numpy(arr.astype(np.float32)) 
                y = torch.tensor(label, dtype=torch.long)
                self.data.append(x)
                self.labels.append(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
