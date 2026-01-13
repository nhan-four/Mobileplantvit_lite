#!/usr/bin/env python3
"""
Script để chia dataset thành train/val/test với tỷ lệ 70-15-15.

Usage:
    python split_dataset.py --source_dir /path/to/source --output_dir /path/to/output [--seed 42]
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def get_image_files(directory: Path) -> List[Path]:
    """Lấy tất cả các file ảnh từ directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(image_files)


def split_class_folder(
    source_dir: Path,
    output_base: Path,
    class_name: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Chia một class folder thành train/val/test."""
    source_class_dir = source_dir / class_name
    
    if not source_class_dir.exists() or not source_class_dir.is_dir():
        print(f"Warning: {source_class_dir} không tồn tại hoặc không phải folder, bỏ qua.")
        return
    
    # Lấy tất cả file ảnh
    image_files = get_image_files(source_class_dir)
    
    if len(image_files) == 0:
        print(f"Warning: Không tìm thấy ảnh nào trong {source_class_dir}, bỏ qua.")
        return
    
    # Shuffle với seed cố định
    rng = np.random.RandomState(seed)
    indices = np.arange(len(image_files))
    rng.shuffle(indices)
    shuffled_files = [image_files[i] for i in indices]
    
    # Tính số lượng cho mỗi split
    n_total = len(shuffled_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # n_test = n_total - n_train - n_val (phần còn lại)
    
    # Chia files
    train_files = shuffled_files[:n_train]
    val_files = shuffled_files[n_train : n_train + n_val]
    test_files = shuffled_files[n_train + n_val :]
    
    # Tạo output directories
    train_dir = output_base / "train" / class_name
    val_dir = output_base / "val" / class_name
    test_dir = output_base / "test" / class_name
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    def copy_files(files: List[Path], dest_dir: Path, split_name: str) -> None:
        for src_file in files:
            dest_file = dest_dir / src_file.name
            # Nếu file đã tồn tại, thêm suffix
            counter = 1
            while dest_file.exists():
                stem = src_file.stem
                suffix = src_file.suffix
                dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            shutil.copy2(src_file, dest_file)
    
    copy_files(train_files, train_dir, "train")
    copy_files(val_files, val_dir, "val")
    copy_files(test_files, test_dir, "test")
    
    print(
        f"  {class_name}: {n_total} images -> "
        f"train: {len(train_files)} ({len(train_files)/n_total*100:.1f}%), "
        f"val: {len(val_files)} ({len(val_files)/n_total*100:.1f}%), "
        f"test: {len(test_files)} ({len(test_files)/n_total*100:.1f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chia dataset thành train/val/test với tỷ lệ 70-15-15"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Thư mục chứa dataset gốc (có các class folders)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Thư mục output sẽ chứa train/val/test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed để đảm bảo reproducibility (default: 42)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Tỷ lệ train (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Tỷ lệ val (default: 0.15)",
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    if not source_dir.exists():
        raise ValueError(f"Source directory không tồn tại: {source_dir}")
    
    if not source_dir.is_dir():
        raise ValueError(f"Source path không phải là directory: {source_dir}")
    
    # Kiểm tra tỷ lệ
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError(
            f"Tổng train_ratio + val_ratio phải < 1.0, "
            f"hiện tại: {args.train_ratio + args.val_ratio}"
        )
    
    # Lấy tất cả các class folders (bỏ qua files và hidden folders)
    class_dirs = [
        d.name
        for d in source_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    
    if len(class_dirs) == 0:
        raise ValueError(f"Không tìm thấy class folder nào trong {source_dir}")
    
    class_dirs = sorted(class_dirs)
    
    print(f"Tìm thấy {len(class_dirs)} classes:")
    for cls in class_dirs:
        print(f"  - {cls}")
    print()
    
    print(f"Bắt đầu chia dataset...")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Train ratio: {args.train_ratio*100:.0f}%")
    print(f"  Val ratio: {args.val_ratio*100:.0f}%")
    print(f"  Test ratio: {(1 - args.train_ratio - args.val_ratio)*100:.0f}%")
    print(f"  Seed: {args.seed}")
    print()
    
    # Chia từng class
    for class_name in class_dirs:
        split_class_folder(
            source_dir=source_dir,
            output_base=output_dir,
            class_name=class_name,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    
    print()
    print("=" * 60)
    print("Hoàn thành!")
    print(f"Dataset đã được chia vào: {output_dir}")
    print("Cấu trúc:")
    print("  output_dir/")
    print("    train/<class_name>/*.jpg")
    print("    val/<class_name>/*.jpg")
    print("    test/<class_name>/*.jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()

