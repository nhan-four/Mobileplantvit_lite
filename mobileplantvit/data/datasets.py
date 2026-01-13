from __future__ import annotations

import os
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class AlbumentationsImageFolder(Dataset):
    """ImageFolder + Albumentations.

    - Reads images via PIL (RGB).
    - Applies Albumentations transform (if provided).
    - Can optionally return file path for test-time analysis.
    - Handles corrupted images gracefully by skipping them.
    """

    def __init__(self, root: str, transform: Optional[Any] = None, return_path: bool = False) -> None:
        self.base = datasets.ImageFolder(root)
        self.transform = transform
        self.return_path = return_path
        
        # Filter out corrupted images during initialization
        self.valid_indices = []
        self.corrupted_files = []
        
        print(f"Scanning dataset at {root} for corrupted images...")
        for idx in range(len(self.base)):
            path, _ = self.base.samples[idx]
            try:
                # Try to open and verify image
                with Image.open(path) as img:
                    img.verify()
                # Reopen for actual loading (verify() closes the file)
                img = Image.open(path)
                img.load()  # Load the image data
                img.close()
                self.valid_indices.append(idx)
            except Exception as e:
                self.corrupted_files.append((path, str(e)))
        
        if self.corrupted_files:
            print(f"Warning: Found {len(self.corrupted_files)} corrupted images, skipping them.")
            print(f"Total valid images: {len(self.valid_indices)}/{len(self.base)}")
            # Print first few corrupted files for debugging
            for path, error in self.corrupted_files[:5]:
                print(f"  - {path}: {error}")
            if len(self.corrupted_files) > 5:
                print(f"  ... and {len(self.corrupted_files) - 5} more")
        else:
            print(f"All {len(self.base)} images are valid.")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int):
        # Map index to valid index
        actual_index = self.valid_indices[index]
        path, label = self.base.samples[actual_index]
        
        # Load image with error handling
        try:
            image = Image.open(path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            # Fallback: return a black image if still fails (shouldn't happen after filtering)
            warnings.warn(f"Failed to load image {path}: {e}, using black image")
            # Create a dummy image (224x224 RGB black image as fallback)
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.return_path:
            return image, label, path
        return image, label


def create_dataloaders(
    data_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    augment_level: str,
    transforms_factory,
    return_paths_for_test: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train/val/test loaders from the required folder structure."""
    if num_workers < 0:
        num_workers = os.cpu_count() or 4

    train_tf, eval_tf = transforms_factory(img_size, augment_level)

    train_ds = AlbumentationsImageFolder(os.path.join(data_root, "train"), transform=train_tf, return_path=False)
    val_ds = AlbumentationsImageFolder(os.path.join(data_root, "val"), transform=eval_tf, return_path=False)
    test_ds = AlbumentationsImageFolder(
        os.path.join(data_root, "test"),
        transform=eval_tf,
        return_path=return_paths_for_test,
    )

    class_names = train_ds.base.classes

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)

    return train_loader, val_loader, test_loader, class_names
