from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils import opp_sliding_window_w_d, get_sample_weights

DATA_DIR = "./data/oppor_preprocessed"
class data_loader_oppor(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain

    def __len__(self):
        return len(self.samples)
    
    def load_domain_data(domain_idx):
        """ to load all the data from the specific domain
        :param domain_idx:
        :return: X and y data of the entire domain
        """
        saved_filename = 'oppor_domain_' + domain_idx + '_wd.data' # with domain label
        data = np.load(DATA_DIR + saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]

def prep_domains_oppor(SLIDING_WINDOW_LEN=30, SLIDING_WINDOW_STEP=15, batch_size = 64, target_domain = 'S1'):
    """
    Build dataloaders from already preprocessed domain files stored in a
    dedicated directory.
    """


    source_domain_list = ['S1', 'S2', 'S3', 'S4']
    source_domain_list.remove(target_domain)

    source_loaders = []

    # source domains
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)

        # load preprocessed domain file
        domain_file = os.path.join(
            DATA_DIR, f"oppor_domain_{source_domain}_wd.data"
        )
        data = np.load(domain_file, allow_pickle=True)
        x, y, d = data[0]

        # sliding window
        x_win, y_win, d_win = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        y_win = np.asarray(y_win).reshape(-1)
        d_win = np.asarray(d_win).reshape(-1)

        if x_win.ndim == 2:
            # expecting [N, T*F]
            assert x_win.shape[1] == SLIDING_WINDOW_LEN * 77, (
                f"Unexpected flattened shape: {x_win.shape}; "
                f"expected second dim {SLIDING_WINDOW_LEN*77}"
            )
            x_win = x_win.reshape(-1, SLIDING_WINDOW_LEN, 77)
        elif x_win.ndim == 3:
            # expecting [N, T, F]
            assert x_win.shape[1] == SLIDING_WINDOW_LEN and x_win.shape[2] == 77, (
                f"Unexpected window shape: {x_win.shape}"
            )
        else:
            raise ValueError(f"Unexpected x_win shape: {x_win.shape}")

        # weighted sampling for class imbalance
        unique_y, counts_y = np.unique(y_win, return_counts=True)
        weights = 100.0 / torch.Tensor(counts_y)
        weights = weights.double()
        sample_weights = get_sample_weights(y_win, weights)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        dataset = data_loader_oppor(x_win, y_win, d_win)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            sampler=sampler
        )

        print('source_loader batch:', len(loader))
        source_loaders.append(loader)

    # target domain
    print('target_domain:', target_domain)

    domain_file = os.path.join(
        DATA_DIR, f"oppor_domain_{target_domain}_wd.data"
    )
    data = np.load(domain_file, allow_pickle=True)
    x, y, d = data[0]

    x_win, y_win, d_win = opp_sliding_window_w_d(
        x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP
    )
    y_win = np.asarray(y_win).reshape(-1)
    d_win = np.asarray(d_win).reshape(-1)

    x_win = np.asarray(x_win)
    if x_win.ndim == 2:
        # expecting [N, T*F]
        assert x_win.shape[1] == SLIDING_WINDOW_LEN * 77, (
            f"Unexpected flattened shape: {x_win.shape}; "
            f"expected second dim {SLIDING_WINDOW_LEN*77}"
        )
        x_win = x_win.reshape(-1, SLIDING_WINDOW_LEN, 77)
    elif x_win.ndim == 3:
        # expecting [N, T, F]
        assert x_win.shape[1] == SLIDING_WINDOW_LEN and x_win.shape[2] == 77, (
            f"Unexpected window shape: {x_win.shape}"
        )
    else:
        raise ValueError(f"Unexpected x_win shape: {x_win.shape}")

    dataset = data_loader_oppor(x_win, y_win, d_win)
    target_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return source_loaders, target_loader
