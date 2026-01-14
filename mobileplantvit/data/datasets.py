from __future__ import annotations

import os
import warnings
from typing import Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


def _validate_single_image(args: Tuple[int, str]) -> Tuple[int, bool, Optional[str]]:
    """
    Validate a single image (helper function for parallel validation).
    
    Args:
        args: Tuple of (index, image_path)
    
    Returns:
        Tuple of (index, is_valid, error_message)
    """
    idx, path = args
    try:
        # Step 1: Try to open and verify image
        with Image.open(path) as img:
            img.verify()
        
        # Step 2: Reopen for actual loading (verify() closes the file)
        img = Image.open(path)
        img.load()  # Load the image data
        img.close()
        
        # Step 3: Test convert to RGB (this is what __getitem__ will do)
        with Image.open(path) as img:
            img.convert("RGB")
        
        return idx, True, None
    except Exception as e:
        return idx, False, str(e)


class AlbumentationsImageFolder(Dataset):
    """ImageFolder + Albumentations with optimized validation.

    - Reads images via PIL (RGB).
    - Applies Albumentations transform (if provided).
    - Can optionally return file path for test-time analysis.
    - Handles corrupted images gracefully by skipping them.
    - Uses parallel validation with caching for faster initialization.
    """

    def __init__(
        self, 
        root: str, 
        transform: Optional[Any] = None, 
        return_path: bool = False,
        enable_cache: bool = True,
        num_workers: int = 10
    ) -> None:
        self.base = datasets.ImageFolder(root)
        self.transform = transform
        self.return_path = return_path
        self.root = root
        
        # Cache file path
        cache_dir = Path(root).parent / ".validation_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{Path(root).name}_validation_cache.json"
        
        # Filter out corrupted images during initialization
        self.valid_indices = []
        self.corrupted_files = []
        
        # Try to load from cache
        cached_data = None
        if enable_cache and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Verify cache is still valid (same number of images)
                if cached_data.get('total_images') == len(self.base):
                    # Verify a few random samples still exist
                    samples_to_check = min(10, len(self.base))
                    all_exist = all(
                        Path(self.base.samples[i][0]).exists() 
                        for i in range(0, len(self.base), max(1, len(self.base) // samples_to_check))
                    )
                    
                    if all_exist:
                        self.valid_indices = cached_data['valid_indices']
                        self.corrupted_files = [tuple(item) for item in cached_data['corrupted_files']]
                        print(f"✓ Loaded validation cache from {cache_file}")
                        print(f"  Valid images: {len(self.valid_indices)}/{len(self.base)}")
                        if self.corrupted_files:
                            print(f"  Skipping {len(self.corrupted_files)} corrupted images")
                        return
                    else:
                        print(f"⚠ Cache invalid (files changed), re-validating...")
                else:
                    print(f"⚠ Cache invalid (image count mismatch), re-validating...")
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}, re-validating...")
        
        # Perform validation (with parallel processing)
        print(f"🔍 Validating {len(self.base)} images in {root}...")
        
        # Prepare validation tasks
        validation_tasks = [(idx, self.base.samples[idx][0]) for idx in range(len(self.base))]
        
        # Use ThreadPoolExecutor for parallel validation
        valid_count = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_validate_single_image, task): task for task in validation_tasks}
            
            for future in as_completed(futures):
                idx, is_valid, error = future.result()
                if is_valid:
                    self.valid_indices.append(idx)
                    valid_count += 1
                else:
                    path = self.base.samples[idx][0]
                    self.corrupted_files.append((path, error))
                
                # Progress indicator every 1000 images
                if (valid_count + len(self.corrupted_files)) % 1000 == 0:
                    print(f"  Progress: {valid_count + len(self.corrupted_files)}/{len(self.base)} images checked")
        
        # Sort valid_indices to maintain order
        self.valid_indices.sort()
        
        # Print summary
        if self.corrupted_files:
            print(f"⚠ Found {len(self.corrupted_files)} corrupted images, skipping them.")
            print(f"✓ Total valid images: {len(self.valid_indices)}/{len(self.base)}")
            # Print first few corrupted files for debugging
            for path, error in self.corrupted_files[:3]:
                print(f"  - {Path(path).name}: {error}")
            if len(self.corrupted_files) > 3:
                print(f"  ... and {len(self.corrupted_files) - 3} more")
        else:
            print(f"✓ All {len(self.base)} images are valid.")
        
        # Save to cache
        if enable_cache:
            try:
                cache_data = {
                    'total_images': len(self.base),
                    'valid_indices': self.valid_indices,
                    'corrupted_files': self.corrupted_files,
                    'root': root
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
                print(f"💾 Saved validation cache to {cache_file}")
            except Exception as e:
                print(f"⚠ Failed to save cache: {e}")

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
    enable_validation_cache: bool = True,
    validation_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build train/val/test loaders from the required folder structure.
    
    Args:
        data_root: Root directory containing train/val/test folders
        img_size: Image size for transforms
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        augment_level: Augmentation level ('light', 'medium', 'heavy')
        transforms_factory: Function to create transforms
        return_paths_for_test: Whether to return paths in test loader
        enable_validation_cache: Enable caching of validation results
        validation_workers: Number of workers for parallel validation
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    if num_workers < 0:
        num_workers = os.cpu_count() or 4

    train_tf, eval_tf = transforms_factory(img_size, augment_level)

    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    print("\n[1/3] Loading training dataset...")
    train_ds = AlbumentationsImageFolder(
        os.path.join(data_root, "train"), 
        transform=train_tf, 
        return_path=False,
        enable_cache=enable_validation_cache,
        num_workers=validation_workers
    )
    
    print("\n[2/3] Loading validation dataset...")
    val_ds = AlbumentationsImageFolder(
        os.path.join(data_root, "val"), 
        transform=eval_tf, 
        return_path=False,
        enable_cache=enable_validation_cache,
        num_workers=validation_workers
    )
    
    print("\n[3/3] Loading test dataset...")
    test_ds = AlbumentationsImageFolder(
        os.path.join(data_root, "test"),
        transform=eval_tf,
        return_path=return_paths_for_test,
        enable_cache=enable_validation_cache,
        num_workers=validation_workers
    )

    class_names = train_ds.base.classes
    
    print("\n" + "="*80)
    print(f"✓ Dataset loaded successfully!")
    print(f"  Classes: {len(class_names)} ({', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''})")
    print(f"  Train: {len(train_ds)} images")
    print(f"  Val: {len(val_ds)} images")
    print(f"  Test: {len(test_ds)} images")
    print("="*80 + "\n")

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
