from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import List, Sequence, Tuple, Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils import opp_sliding_window_w_d  # last-label windowing


LabelType = Literal["gestures", "locomotion"]

NUM_FEATURES = 77
PREPROCESSED_ROOT = Path(os.getenv("OPP_PREPROCESSED_ROOT", "./data/oppor_preprocessed/"))


def _domain_id_to_int(domain_id: str) -> int:
    m = re.search(r"(\d+)$", domain_id)
    if not m:
        raise ValueError(f"cannot parse numeric suffix from domain_id={domain_id!r}")
    return int(m.group(1)) - 1  # S1->0, ...


def _expected_num_classes(label_type: LabelType) -> int:
    # OPPORTUNITY: gestures are typically 18 classes (including 0 = "null"),
    # locomotion is usually 5 (or 4/5 depending on mapping). Your setup uses 18.
    return 18 if label_type == "gestures" else 5


def _preprocessed_candidates(domain_id: str, label_type: LabelType) -> List[Path]:
    # Make label_type explicit in filenames to avoid accidentally loading the wrong task.
    # Preferred:
    #   oppor_domain_S1_gestures_wd.data
    #   oppor_domain_S1_locomotion_wd.data
    #
    # Fallbacks (less explicit) are still supported, but we validate label ranges later.
    return [
        PREPROCESSED_ROOT / f"oppor_domain_{domain_id}_{label_type}_wd.data",
        PREPROCESSED_ROOT / f"oppor_domain_{domain_id}_wd.data",  # legacy author name
        PREPROCESSED_ROOT / f"{domain_id}_wd.data",
    ]


def load_domain_wd(domain_id: str, label_type: LabelType) -> Tuple[np.ndarray, np.ndarray]:
    file_path: Optional[Path] = None
    for cand in _preprocessed_candidates(domain_id, label_type):
        if cand.is_file():
            file_path = cand
            break

    if file_path is None:
        raise FileNotFoundError(
            f"Could not find preprocessed .data for domain {domain_id} (label_type={label_type}) in {PREPROCESSED_ROOT}.\n"
            f"Expected e.g. oppor_domain_{domain_id}_{label_type}_wd.data"
        )

    with file_path.open("rb") as f:
        obj = pickle.load(f)

    parts = obj if isinstance(obj, list) else [obj]

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for part in parts:
        if not isinstance(part, tuple) or len(part) < 2:
            raise TypeError(f"Unexpected pickle part structure in {file_path}: {type(part)}")

        X = np.asarray(part[0], dtype=np.float32)
        y = np.asarray(part[1])

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        y = y.astype(np.int64)

        if X.ndim != 2 or X.shape[1] != NUM_FEATURES:
            raise ValueError(f"Expected X shape (N,{NUM_FEATURES}), got {X.shape} from {file_path}")

        xs.append(X)
        ys.append(y)

    X_all = np.vstack(xs).astype(np.float32)
    y_all = np.concatenate(ys).astype(np.int64)

    # Basic sanity checks: prevent silent task mismatches
    if np.isnan(X_all).any():
        raise ValueError(f"Found NaNs in X for {domain_id} ({label_type}). Preprocessing mismatch/corruption.")

    # Validate label range against the requested task
    exp_classes = _expected_num_classes(label_type)
    y_min = int(y_all.min()) if y_all.size else 0
    y_max = int(y_all.max()) if y_all.size else 0

    # Gestures typically include 0 as "null". Locomotion often includes 0 too, depending on mapping.
    if y_min < 0 or y_max >= exp_classes:
        raise ValueError(
            f"Label range mismatch for {domain_id}: y in [{y_min}, {y_max}] but label_type={label_type} "
            f"expects < {exp_classes} classes.\n"
            f"This usually means you loaded the wrong preprocessed file (gestures vs locomotion) or the mapping differs."
        )

    return X_all, y_all


class OpportunityAuthorWindowDataset(Dataset):
    def __init__(
        self,
        domain_ids: Sequence[str],
        *,
        label_type: LabelType,
        window_size: int,
        step_size: int,
        flatten: bool,
        expected_x_dim: Optional[int] = None,  # e.g., 30*77
    ):
        xs, ys, ds = [], [], []

        for dom in domain_ids:
            X, y = load_domain_wd(dom, label_type=label_type)
            d = np.full((len(y),), _domain_id_to_int(dom), dtype=np.int64)

            x_win, y_win, d_win = opp_sliding_window_w_d(
                X, y, d,
                window_size=window_size,
                step_size=step_size,
            )

            # Enforce explicit expectations
            if np.isnan(x_win).any():
                raise ValueError(f"NaNs after windowing for domain {dom}. Check normalization / interpolation.")

            if flatten:
                # expected shape: (Nwin, window_size*NUM_FEATURES)
                if x_win.ndim != 2 or x_win.shape[1] != window_size * NUM_FEATURES:
                    raise ValueError(
                        f"Windowed X has shape {x_win.shape}, expected (N,{window_size*NUM_FEATURES})."
                    )
                if expected_x_dim is not None and x_win.shape[1] != expected_x_dim:
                    raise ValueError(
                        f"x_dim mismatch: loader produced {x_win.shape[1]} but model expects {expected_x_dim}."
                    )
            else:
                # optional non-flattened support if you ever need it
                if x_win.ndim != 3 or x_win.shape[1:] != (window_size, NUM_FEATURES):
                    raise ValueError(
                        f"Windowed X has shape {x_win.shape}, expected (N,{window_size},{NUM_FEATURES})."
                    )

            xs.append(x_win)
            ys.append(y_win.astype(np.int64).reshape(-1))
            ds.append(d_win.astype(np.int64).reshape(-1))

        self.x = torch.from_numpy(np.vstack(xs)).float()
        self.y = torch.from_numpy(np.concatenate(ys)).long()
        self.d = torch.from_numpy(np.concatenate(ds)).long()

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.d[idx]


def build_opportunity_loader(
    *,
    domain_ids: Sequence[str],
    label_type: LabelType = "gestures",
    batch_size: int = 64,
    balanced: bool = False,
    shuffle: bool = False,
    window_size: int = 30,
    step_size: int = 15,
    flatten: bool = True,
    expected_x_dim: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    dataset = OpportunityAuthorWindowDataset(
        domain_ids=domain_ids,
        label_type=label_type,
        window_size=window_size,
        step_size=step_size,
        flatten=flatten,
        expected_x_dim=expected_x_dim,
    )

    if balanced:
        y = dataset.y
        num_classes = int(y.max().item()) + 1 if y.numel() else 1
        counts = torch.bincount(y, minlength=num_classes).float().clamp_min(1.0)
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
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
