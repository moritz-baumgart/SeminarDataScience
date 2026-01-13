from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pandas import Series

from utils import opp_sliding_window_w_d


# Config 
NUM_FEATURES = 77

DATA_ROOT = Path("./data/OpportunityUCIDataset/dataset/")

CACHE_DIR = Path("./data/oppor/cache/")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 30
STEP_SIZE   = 15

#WINDOW_SIZE = int(os.getenv("OPP_WINDOW_SIZE", "24"))
#STEP_SIZE   = int(os.getenv("OPP_STEP_SIZE", "12")) 


# Dataset
class OpportunityDataset(Dataset):
    """
    OPPORTUNITY Dataset (Autor-Preprocessing + Windowing)
    Returns:
        x : FloatTensor [N, WINDOW_SIZE * 77]
        y : LongTensor  [N]
        d : LongTensor  [N]  (S1..S4 -> 0..3)
    """

    def __init__(self, domain_ids: List[str], label_type: str = "gestures"):
        x, y, d = self._load_domains(domain_ids, label_type)

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.d = torch.from_numpy(d).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.d[idx]

    def _load_domains(self, domain_ids: List[str], label_type: str):
        xs, ys, ds = [], [], []
        for dom in domain_ids:
            x, y, d = load_domain(dom, label_type, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
            xs.append(x)
            ys.append(y)
            ds.append(d)

        x_all = np.vstack(xs).astype(np.float32)
        y_all = np.concatenate(ys).astype(np.int64)
        d_all = np.concatenate(ds).astype(np.int64)
        return x_all, y_all, d_all


# Domain Loading + Preprocessing
def _domain_file_list(domain_id: str) -> List[Path]:
    """
    """
    names = [
        f"{domain_id}-Drill.dat",
        f"{domain_id}-ADL1.dat",
        f"{domain_id}-ADL2.dat",
        f"{domain_id}-ADL3.dat",
        f"{domain_id}-ADL4.dat",
        f"{domain_id}-ADL5.dat",
    ]
    files = [DATA_ROOT / n for n in names if (DATA_ROOT / n).is_file()]
    if len(files) == 0:
        files = sorted([p for p in DATA_ROOT.iterdir() if p.name.startswith(domain_id) and p.suffix == ".dat"])
    return files


def load_domain(
    domain_id: str,
    label_type: str,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    cache_path = CACHE_DIR / f"oppor_{domain_id}_{label_type}_w{window_size}_s{step_size}.npz"
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=False)
        return data["x"], data["y"], data["d"]

    files = _domain_file_list(domain_id)
    if len(files) == 0:
        raise FileNotFoundError(f"No .dat files found for domain {domain_id} under {DATA_ROOT}")

    data_x_parts = []
    data_y_parts = []

    for fp in files:
        raw = np.loadtxt(fp)
        x, y = process_dataset_file(raw, label_type)
        data_x_parts.append(x)
        data_y_parts.append(y)

    x = np.vstack(data_x_parts)
    y = np.concatenate(data_y_parts).astype(np.int64)

    d = np.full(shape=(len(y),), fill_value=int(domain_id[-1]) - 1, dtype=np.int64)

    # Sliding Window + Flatten (WINDOW_SIZE*77)
    x_win, y_win, d_win = opp_sliding_window_w_d(
        x, y, d,
        window_size=window_size,
        step_size=step_size,
    )
    y_win = y_win.reshape(-1).astype(np.int64)
    d_win = d_win.reshape(-1).astype(np.int64)

    np.savez_compressed(cache_path, x=x_win, y=y_win, d=d_win)
    return x_win, y_win, d_win


def process_dataset_file(data: np.ndarray, label: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    data = select_columns_opp(data)
    data_x, data_y = divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label).astype(int)

    data_x = np.array([Series(col).interpolate().to_numpy() for col in data_x.T]).T
    data_x[np.isnan(data_x)] = 0.0

    data_x = normalize(data_x)
    return data_x.astype(np.float32), data_y.astype(np.int64)


def select_columns_opp(data: np.ndarray) -> np.ndarray:
    """
    """
    features_delete = np.arange(0, 37)
    features_delete = np.concatenate([features_delete, np.arange(46, 50)])
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])

    return np.delete(data, features_delete, axis=1)


def divide_x_y(data: np.ndarray, label: str) -> Tuple[np.ndarray, np.ndarray]:
    data_x = data[:, :NUM_FEATURES]

    if label not in {"locomotion", "gestures"}:
        raise ValueError(f"Invalid label_type '{label}', expected 'gestures' or 'locomotion'")

    if label == "locomotion":
        data_y = data[:, NUM_FEATURES]        # locomotion label
    else:
        data_y = data[:, NUM_FEATURES + 1]    # gestures label

    return data_x, data_y


def adjust_idx_labels(data_y: np.ndarray, label: str) -> np.ndarray:
    """
    """
    y = data_y.copy()

    if label == "locomotion":
        y[y == 4] = 3
        y[y == 5] = 4
        return y

    mapping = {
        406516: 1,
        406517: 2,
        404516: 3,
        404517: 4,
        406520: 5,
        404520: 6,
        406505: 7,
        404505: 8,
        406519: 9,
        404519: 10,
        406511: 11,
        404511: 12,
        406508: 13,
        404508: 14,
        408512: 15,
        407521: 16,
        405506: 17,
    }
    for k, v in mapping.items():
        y[y == k] = v
    return y


def normalize(x: np.ndarray) -> np.ndarray:
    """
    """
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-6
    return (x - mean) / std


def build_opportunity_loader(
    domain_ids,
    batch_size: int = 64,
    shuffle: bool = True,
    label_type: str = "gestures",
):
    """

    """
    dataset = OpportunityDataset(domain_ids=domain_ids, label_type=label_type)

    if shuffle:
        y = dataset.y
        num_classes = int(torch.max(y).item()) + 1
        counts = torch.bincount(y, minlength=num_classes).float()
        counts = torch.clamp(counts, min=1.0)

        class_w = 1.0 / counts
        sample_w = class_w[y].double()

        sampler = WeightedRandomSampler(
            weights=sample_w,
            num_samples=len(sample_w),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
