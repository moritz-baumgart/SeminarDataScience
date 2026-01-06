# encoding=utf-8
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pandas import Series
from utils import opp_sliding_window_w_d
# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_FEATURES = 77
DATA_ROOT = "./data/OpportunityUCIDataset/dataset/"
WINDOW_SIZE = 24
STEP_SIZE = 12
# --------------------------------------------------
# Dataset
# --------------------------------------------------

class OpportunityDataset(Dataset):
    """
    Minimal OPPORTUNITY Dataset for GILE

    Returns:
        x : FloatTensor [B, 77]
        y : LongTensor  [B]   activity label
        d : LongTensor  [B]   domain label (S1–S4 -> 0–3)
    """

    def __init__(self, domain_ids, label_type="gestures"):
        self.x, self.y, self.d = self._load_domains(domain_ids, label_type)

        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).long()
        self.d = torch.from_numpy(self.d).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.d[idx]

    def _load_domains(self, domain_ids, label_type):
        xs, ys, ds = [], [], []

        for domain_id in domain_ids:
            x, y, d = load_domain(domain_id, label_type)
            xs.append(x)
            ys.append(y)
            ds.append(d)

        return (
            np.vstack(xs),
            np.concatenate(ys),
            np.concatenate(ds),
        )

# --------------------------------------------------
# Domain loading
# --------------------------------------------------

def load_domain(domain_id, label_type):
    files = [
        f for f in os.listdir(DATA_ROOT)
        if f.startswith(domain_id)
    ]

    data_x, data_y = [], []

    for f in files:
        path = os.path.join(DATA_ROOT, f)
        raw = np.loadtxt(path)
        x, y = process_dataset_file(raw, label_type)
        data_x.append(x)
        data_y.append(y)

    x = np.vstack(data_x)
    y = np.concatenate(data_y)
    # domain label: S1 -> 0, S2 -> 1, ...
    d = np.full(len(y), int(domain_id[-1]) - 1, dtype=int)

    # sliding window
    x, y, d = opp_sliding_window_w_d(x, y, d, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    y = y.reshape(-1) 
    d = d.reshape(-1)
    return x, y, d

# --------------------------------------------------
# Preprocessing pipeline
# --------------------------------------------------

def process_dataset_file(data, label):
    data = select_columns_opp(data)
    data_x, data_y = divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)

    # interpolate missing values
    data_x = np.array([Series(col).interpolate() for col in data_x.T]).T
    data_x[np.isnan(data_x)] = 0

    data_x = normalize(data_x)

    return data_x, data_y.astype(int)

def select_columns_opp(data):
    features_delete = np.arange(0, 37)
    features_delete = np.concatenate([features_delete, np.arange(46, 50)])
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])

    data = np.delete(data, features_delete, axis=1)
    return data

def divide_x_y(data, label):
    data_x = data[:, :NUM_FEATURES]

    if label == "locomotion":
        data_y = data[:, NUM_FEATURES]
    elif label == "gestures":
        data_y = data[:, NUM_FEATURES + 1]
    else:
        raise ValueError(f"Unknown label type: {label}")

    return data_x, data_y

def adjust_idx_labels(data_y, label):
    if label == "locomotion":
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4

    elif label == "gestures":
        mapping = {
            406516: 1, 406517: 2, 404516: 3, 404517: 4,
            406520: 5, 404520: 6, 406505: 7, 404505: 8,
            406519: 9, 404519: 10, 406511: 11, 404511: 12,
            406508: 13, 404508: 14, 408512: 15,
            407521: 16, 405506: 17,
        }
        for k, v in mapping.items():
            data_y[data_y == k] = v

    return data_y

def normalize(x):
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-6
    x = (x - mean) / std
    return x

# --------------------------------------------------
# Convenience loader builders
# --------------------------------------------------

def build_opportunity_loader(
    domain_ids,
    batch_size=64,
    shuffle=True,
    label_type="gestures",
):
    dataset = OpportunityDataset(domain_ids, label_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# --------------------------------------------------
# Example usage (LOSO)
# --------------------------------------------------

if __name__ == "__main__":
    train_loader = build_opportunity_loader(
        domain_ids=["S1", "S2", "S3"],
        batch_size=64,
        shuffle=True,
    )

    test_loader = build_opportunity_loader(
        domain_ids=["S4"],
        batch_size=64,
        shuffle=False,
    )

    x, y, d = next(iter(train_loader))
    print(x.shape, y.shape, d.shape)
