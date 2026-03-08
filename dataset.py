"""
src/dataset.py
--------------
PatchCamelyon (PCam) dataset loader with heavy augmentation pipeline.
Also includes a generic patch dataset for custom WSI tiles.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# PatchCamelyon Dataset
# ──────────────────────────────────────────────

class PCamDataset(Dataset):
    """
    PatchCamelyon (PCam) Dataset.

    Downloads from: https://github.com/basveeling/pcam
    HDF5 format: x.h5 (images, uint8, NHWC) and y.h5 (labels, int, N111)

    Args:
        data_dir  : path to directory containing x.h5 and y.h5
        split     : 'train', 'valid', or 'test'
        transform : albumentations transform pipeline
        img_size  : resize target (default 224 for EfficientNet)
    """

    SPLIT_FILES = {
        'train': ('camelyonpatch_level_2_split_train_x.h5',
                  'camelyonpatch_level_2_split_train_y.h5'),
        'valid': ('camelyonpatch_level_2_split_valid_x.h5',
                  'camelyonpatch_level_2_split_valid_y.h5'),
        'test':  ('camelyonpatch_level_2_split_test_x.h5',
                  'camelyonpatch_level_2_split_test_y.h5'),
    }

    def __init__(self, data_dir: str, split: str = 'train',
                 transform=None, img_size: int = 224):
        assert split in self.SPLIT_FILES, f"split must be one of {list(self.SPLIT_FILES)}"

        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform or self._default_transform(split)

        x_file, y_file = self.SPLIT_FILES[split]
        x_path = os.path.join(data_dir, x_file)
        y_path = os.path.join(data_dir, y_file)

        if not os.path.exists(x_path):
            raise FileNotFoundError(
                f"PCam HDF5 not found at {x_path}.\n"
                "Download from: https://github.com/basveeling/pcam"
            )

        # Load into memory (PCam is ~7GB total; use memory-mapped for large splits)
        with h5py.File(x_path, 'r') as f:
            self.images = f['x'][:]          # (N, 96, 96, 3) uint8
        with h5py.File(y_path, 'r') as f:
            self.labels = f['y'][:].squeeze().astype(np.int64)  # (N,)

        pos = self.labels.sum()
        logger.info(f"PCam [{split}]: {len(self.labels):,} samples | "
                    f"Positive: {pos:,} ({pos/len(self.labels):.1%})")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]   # (96, 96, 3) uint8 numpy
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']   # Tensor after ToTensorV2

        return img, torch.tensor(label, dtype=torch.float32)

    @staticmethod
    def _default_transform(split: str, img_size: int = 224):
        """Default albumentations transforms per split."""
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        if split == 'train':
            return A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.7),
                A.ColorJitter(brightness=0.3, contrast=0.3,
                              saturation=0.2, hue=0.1, p=0.6),
                A.ElasticTransform(alpha=120, sigma=8, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.ToGray(p=0.1),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])

    def get_sampler(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler for class-balanced batches."""
        class_counts = np.bincount(self.labels)
        weights = 1.0 / class_counts
        sample_weights = torch.tensor(weights[self.labels], dtype=torch.double)
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ──────────────────────────────────────────────
# Generic Patch Dataset (for custom WSI tiles)
# ──────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    Loads image patches from a directory (or list of paths) for inference.
    Used for sliding-window WSI inference.

    directory structure:
        patches/
            slide_001_x0_y0.png
            slide_001_x0_y256.png
            ...
    """

    def __init__(self, patch_paths: list, labels: list = None,
                 img_size: int = 224, transform=None):
        self.patch_paths = patch_paths
        self.labels = labels
        self.img_size = img_size

        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        self.transform = transform or A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.patch_paths[idx]).convert('RGB'))
        augmented = self.transform(image=img)
        img_tensor = augmented['image']

        if self.labels is not None:
            return img_tensor, torch.tensor(self.labels[idx], dtype=torch.float32)
        return img_tensor


# ──────────────────────────────────────────────
# Hard Negative Dataset (for HNM training)
# ──────────────────────────────────────────────

class HardNegativeDataset(Dataset):
    """
    Wraps a base dataset and oversamples hard examples (false positives
    and false negatives identified after a training epoch).
    """

    def __init__(self, base_dataset: Dataset, hard_indices: list,
                 oversample_factor: int = 3):
        self.base_dataset = base_dataset
        # Concatenate original indices + repeated hard indices
        hard_repeated = hard_indices * oversample_factor
        all_indices = list(range(len(base_dataset))) + hard_repeated
        self.all_indices = all_indices
        logger.info(f"HNM dataset: {len(base_dataset):,} base + "
                    f"{len(hard_repeated):,} hard examples = {len(all_indices):,} total")

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.all_indices[idx]]


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────

def get_dataloaders(data_dir: str, img_size: int = 224,
                    batch_size: int = 64, num_workers: int = 4,
                    use_sampler: bool = True):
    """
    Build train / val / test DataLoaders for PCam.
    """
    train_ds = PCamDataset(data_dir, split='train', img_size=img_size)
    val_ds   = PCamDataset(data_dir, split='valid', img_size=img_size)
    test_ds  = PCamDataset(data_dir, split='test',  img_size=img_size)

    sampler = train_ds.get_sampler() if use_sampler else None
    shuffle = not use_sampler

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Quick smoke test with synthetic data
    print("Testing with synthetic dummy data...")

    dummy_images = np.random.randint(0, 255, (100, 96, 96, 3), dtype=np.uint8)
    dummy_labels = np.random.randint(0, 2, (100,), dtype=np.int64)

    class DummyPCam(PCamDataset):
        def __init__(self):
            self.images = dummy_images
            self.labels = dummy_labels
            self.transform = PCamDataset._default_transform('train')

    ds = DummyPCam()
    img, label = ds[0]
    print(f"Image shape: {img.shape} | Label: {label}")
    print("Dataset test passed ✅")
