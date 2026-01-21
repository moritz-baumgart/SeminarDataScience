from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import get_sample_weights

UCIHAR_DATA_DIR = "./data/ucihar_preprocessed"  # adjust to your folder


class data_loader_ucihar(Dataset):
    def __init__(self, samples, labels, domains, t):
        self.samples = samples
        self.labels = labels
        self.domains = domains
        self.T = t

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        sample = self.T(sample)
        return np.transpose(sample, (1, 0, 2)), target, domain

    def __len__(self):
        return len(self.samples)


def prep_domains_ucihar_preprocessed(
    batch_size: int = 64,
    target_domain: str = "0",
    data_dir: str = UCIHAR_DATA_DIR,
):
    """
    Build dataloaders from already-preprocessed domain files:
      ucihar_domain_{domain}_wd.data
    Each file must contain obj = [(X, y, d)] in the author's format.
    """

    source_domain_list = ["0", "1", "2", "3", "4"]
    source_domain_list.remove(target_domain)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0, 0, 0, 0, 0, 0, 0, 0, 0),
                std=(1, 1, 1, 1, 1, 1, 1, 1, 1),
            ),
        ]
    )

    source_loaders = []

    # source domains
    for source_domain in source_domain_list:
        print("source_domain:", source_domain)

        domain_file = os.path.join(data_dir, f"ucihar_domain_{source_domain}_wd.data")
        data = np.load(domain_file, allow_pickle=True)
        x, y, d = data[0]  # (X, y, d) stored as obj=[(X,y,d)]

        # match author tensor layout
        x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))  # [N,128,1,9]
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

        dataset = data_loader_ucihar(x, y, d, transform)
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

    domain_file = os.path.join(data_dir, f"ucihar_domain_{target_domain}_wd.data")
    data = np.load(domain_file, allow_pickle=True)
    x, y, d = data[0]

    x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
    y = np.asarray(y).reshape(-1)
    d = np.asarray(d).reshape(-1)

    dataset = data_loader_ucihar(x, y, d, transform)
    target_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("target_loader batch:", len(target_loader))
    return source_loaders, target_loader
