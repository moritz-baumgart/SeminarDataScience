from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import get_sample_weights

SHAR_DATA_DIR = "./data/shar_preprocessed"  # adjust to your folder


class data_loader_shar(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain

    def __len__(self):
        return len(self.samples)


def prep_domains_shar_preprocessed(
    batch_size: int = 64,
    target_domain: str = "1",
    data_dir: str = SHAR_DATA_DIR,
):
    """
    Build dataloaders from already-preprocessed domain files:
      shar_domain_{domain}_wd.data
    Each file must contain obj = [(X, y, d)] in the author's format.
    """

    source_domain_list = ["1", "2", "3", "5"]
    source_domain_list.remove(target_domain)

    source_loaders = []

    # source domains
    for source_domain in source_domain_list:
        print("source_domain:", source_domain)

        domain_file = os.path.join(data_dir, f"shar_domain_{source_domain}_wd.data")
        data = np.load(domain_file, allow_pickle=True)
        x, y, d = data[0]

        x = x.reshape(-1, 151, 3)
        y = np.asarray(y).reshape(-1)
        d = np.asarray(d).reshape(-1)

        # weighted sampling for class imbalance
        unique_y, counts_y = np.unique(y, return_counts=True)
        weights = 100.0 / torch.Tensor(counts_y)
        weights = weights.double()
        sample_weights = get_sample_weights(y, weights)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        dataset = data_loader_shar(x, y, d)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            sampler=sampler,
        )

        print("source_loader batch:", len(loader))
        source_loaders.append(loader)

    # target domain
    print("target_domain:", target_domain)

    domain_file = os.path.join(data_dir, f"shar_domain_{target_domain}_wd.data")
    data = np.load(domain_file, allow_pickle=True)
    x, y, d = data[0]

    x = x.reshape(-1, 151, 3)
    y = np.asarray(y).reshape(-1)
    d = np.asarray(d).reshape(-1)

    dataset = data_loader_shar(x, y, d)
    target_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("target_loader batch:", len(target_loader))
    return source_loaders, target_loader
