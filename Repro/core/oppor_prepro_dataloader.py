from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


NUM_FEATURES = 77

# where the authors' *.data files live
PREPROCESSED_ROOT = Path(os.getenv("OPP_PREPROCESSED_ROOT", "./data/oppor_preprocessed/"))

CACHE_DIR = Path("./data/oppor/cache/")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 30
STEP_SIZE   = 15


def _preprocessed_candidates(domain_id: str, label_type: str) -> List[Path]:
    return [
        PREPROCESSED_ROOT / f"oppor_domain_{domain_id}_wd.data",
        PREPROCESSED_ROOT / f"oppor_domain_{domain_id}_{label_type}_wd.data",
        PREPROCESSED_ROOT / f"{domain_id}_wd.data",
        PREPROCESSED_ROOT / f"{domain_id}_{label_type}_wd.data",
    ]


def _load_preprocessed_series(domain_id: str, label_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads the authors' pickle format.
    Observed format (your S1 file):
        obj is a list with 1 element, that element is a tuple (X, y, d)
        X: (N, 77) float
        y: (N, 1)  float/int labels
        d: (N, 1)  int domain ids (often redundant)
    We also handle: obj being a tuple directly, or list of multiple tuples.
    """
    file_path: Optional[Path] = None
    for cand in _preprocessed_candidates(domain_id, label_type):
        if cand.is_file():
            file_path = cand
            break
    if file_path is None:
        raise FileNotFoundError(
            f"no preprocessed .data found for {domain_id} in {PREPROCESSED_ROOT}. "
            f"expected something like oppor_domain_{domain_id}_wd.data"
        )

    with file_path.open("rb") as f:
        obj = pickle.load(f)

    parts: List[tuple]
    if isinstance(obj, list):
        parts = obj
    elif isinstance(obj, tuple):
        parts = [obj]
    else:
        raise TypeError(f"unexpected pickle root type: {type(obj)}")

    xs, ys, ds = [], [], []
    for part in parts:
        if not isinstance(part, tuple) or len(part) < 2:
            raise TypeError(f"unexpected part in pickle: {type(part)} / len={getattr(part,'__len__',None)}")

        X = np.asarray(part[0])
        y = np.asarray(part[1])
        d = np.asarray(part[2]) if len(part) >= 3 else None

        # normalize shapes
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if d is None:
            d = np.full((len(y),), int(domain_id[-1]) - 1, dtype=np.int64)
        else:
            d = d.reshape(-1) if d.ndim > 1 else d

        xs.append(X)
        ys.append(y)
        ds.append(d)

    X_all = np.vstack(xs).astype(np.float32)
    y_all = np.concatenate(ys).astype(np.int64)
    d_all = np.concatenate(ds).astype(np.int64)

    # if the file has a constant domain label, enforce it anyway
    d_all[:] = int(domain_id[-1]) - 1
    return X_all, y_all, d_all


def _windowize_opportunity(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    *,
    window_size: int,
    step_size: int,
    drop_null: bool = False,
    label_strategy: str = "last",   # "mode" or "last"
    flatten: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x: (N, 77), y: (N,), d: (N,)
    returns:
      x_win: (M, window*77) if flatten else (M, 1, window, 77)
      y_win: (M,)
      d_win: (M,)
    """
    n = len(y)
    if n < window_size:
        raise ValueError(f"sequence shorter than window_size: {n} < {window_size}")

    x_out, y_out, d_out = [], [], []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        xw = x[start:end]                 # (W, 77)
        yw = y[start:end].astype(np.int64)

        if label_strategy == "last":
            lab = int(yw[-1])
        elif label_strategy == "mode":
            lab = int(np.bincount(yw).argmax()) if yw.size else 0
        else:
            raise ValueError("label_strategy must be 'mode' or 'last'")

        if drop_null and lab == 0:
            continue

        x_out.append(xw)
        y_out.append(lab)
        d_out.append(int(d[end - 1]))

    xw = np.stack(x_out, axis=0).astype(np.float32)   # (M, W, 77)
    yw = np.asarray(y_out, dtype=np.int64)
    dw = np.asarray(d_out, dtype=np.int64)

    if flatten:
        xw = xw.reshape(xw.shape[0], -1)              # (M, W*77)
    else:
        xw = xw[:, None, :, :]                        # (M, 1, W, 77)

    return xw, yw, dw


def load_domain(
    domain_id: str,
    label_type: str,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    loads authors preprocessed .data, then windowing + caching
    """
    cache_path = CACHE_DIR / f"oppor_pre_{domain_id}_{label_type}_w{window_size}_s{step_size}.npz"
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=False)
        return data["x"], data["y"], data["d"]

    x_raw, y_raw, d_raw = _load_preprocessed_series(domain_id, label_type)

    x_win, y_win, d_win = _windowize_opportunity(
        x_raw, y_raw, d_raw,
        window_size=window_size,
        step_size=step_size,
        drop_null=True,
        label_strategy="mode",
        flatten=True,
    )

    np.savez_compressed(cache_path, x=x_win, y=y_win, d=d_win)
    return x_win, y_win, d_win


class OpportunityDataset(Dataset):
    def __init__(self, domain_ids: List[str], label_type: str = "gestures"):
        xs, ys, ds = [], [], []
        for dom in domain_ids:
            x, y, d = load_domain(dom, label_type, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
            xs.append(x); ys.append(y); ds.append(d)

        x_all = np.vstack(xs).astype(np.float32)
        y_all = np.concatenate(ys).astype(np.int64)
        d_all = np.concatenate(ds).astype(np.int64)

        self.x = torch.from_numpy(x_all).float()
        self.y = torch.from_numpy(y_all).long()
        self.d = torch.from_numpy(d_all).long()

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.d[idx]


def build_opportunity_loader(
    domain_ids,
    batch_size: int = 64,
    shuffle: bool = True,
    label_type: str = "gestures",
):
    dataset = OpportunityDataset(domain_ids=domain_ids, label_type=label_type)

    if shuffle:
        y = dataset.y
        num_classes = int(torch.max(y).item()) + 1
        counts = torch.bincount(y, minlength=num_classes).float().clamp_min(1.0)
        class_w = 100.0 / counts
        sample_w = class_w[y].double()

        sampler = WeightedRandomSampler(
            weights=sample_w,
            num_samples=len(sample_w),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
