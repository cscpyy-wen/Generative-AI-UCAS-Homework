"""Official MNIST, static threshold binarization, and the fixed split."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class MNISTData:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    checksums: dict

    def loader(self, split, batch_size, device, shuffle=False, seed=20260921):
        dataset = getattr(self, split)
        options = dict(batch_size=batch_size, num_workers=0,
                       pin_memory=torch.device(device).type == "cuda", shuffle=shuffle)
        if shuffle:
            options["generator"] = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, **options)

    def save_protocol(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory/"split_indices.npz",
                            train=self.train_indices.numpy(), val=self.validation_indices.numpy())
        (directory/"dataset_checksums.json").write_text(json.dumps(self.checksums, indent=2))


def load_mnist(data_root="data", *, download=True, threshold=128, split_seed=20260920):
    # Import lazily: model-only smoke tests do not require torchvision operators.
    from torchvision.datasets import MNIST
    MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
                     "http://yann.lecun.com/exdb/mnist/"]
    raw_train = MNIST(str(data_root), train=True, download=download)
    raw_test = MNIST(str(data_root), train=False, download=download)
    checksums = {}
    for name, expected in MNIST.resources:
        compressed = Path(raw_train.raw_folder)/name
        if compressed.exists():
            actual = hashlib.md5(compressed.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("MNIST checksum mismatch: " + name)
            checksums[name] = actual
        else:
            extracted = compressed.with_suffix("")
            if not extracted.exists():
                raise FileNotFoundError(str(extracted))
            checksums[extracted.name + " (sha256)"] = hashlib.sha256(extracted.read_bytes()).hexdigest()
    permutation = torch.randperm(60000, generator=torch.Generator().manual_seed(split_seed))
    train_index, val_index = permutation[:55000], permutation[55000:]
    train_pixels = (raw_train.data >= threshold).float().unsqueeze(1)
    test_pixels = (raw_test.data >= threshold).float().unsqueeze(1)
    return MNISTData(
        TensorDataset(train_pixels[train_index], raw_train.targets[train_index]),
        TensorDataset(train_pixels[val_index], raw_train.targets[val_index]),
        TensorDataset(test_pixels, raw_test.targets),
        train_index, val_index, checksums)
