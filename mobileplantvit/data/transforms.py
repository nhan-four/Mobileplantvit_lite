from __future__ import annotations

from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size: int, augment_level: str = "paper"):
    """Return (train_transform, eval_transform) with Albumentations.
    
    Augment levels:
        - none: No augmentation (resize + normalize only)
        - light: Basic augmentation (flip + brightness/contrast)
        - medium: Moderate augmentation
        - strong: Heavy augmentation (all transforms)
        - paper: Augmentation following paper specifications (default, recommended)
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    level = (augment_level or "paper").lower().strip()

    if level == "paper":
        # Augmentation following paper specifications
        train_tf = A.Compose([
            A.Resize(height=img_size, width=img_size),
            
            # (1) Horizontal flip 50%
            A.HorizontalFlip(p=0.5),
            
            # (2) Random rotate 90° 50%
            A.RandomRotate90(p=0.5),
            
            # (3) Shift/Scale/Rotate 50% (0.05, 0.05, 30°)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=30,
                border_mode=0,
                p=0.5
            ),
            
            # (4) Random Gamma 20%
            A.RandomGamma(p=0.2),
            
            # (5) Brightness/Contrast 30%
            A.RandomBrightnessContrast(p=0.3),
            
            # (6) RGB Shift 30% (limits=15)
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            
            # (7) CLAHE 30% (clip_limit=4.0)
            A.CLAHE(clip_limit=4.0, p=0.3),
            
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif level == "none":
        train_tf = A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif level == "light":
        train_tf = A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif level == "medium":
        train_tf = A.Compose(
            [
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.08, 0.08), rotate=(-30, 30), p=0.5),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(0.2, 0.2, p=1.0),
                        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                        A.CLAHE(clip_limit=3.0, p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.MotionBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.2,
                ),
                A.CoarseDropout(
                    max_holes=6,
                    max_height=img_size // 10,
                    max_width=img_size // 10,
                    fill_value=0,
                    p=0.25,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        train_tf = A.Compose(
            [
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.85, 1.15),
                    translate_percent=(-0.15, 0.15),
                    rotate=(-45, 45),
                    shear=(-10, 10),
                    p=0.6,
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1.0),
                    ],
                    p=0.25,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(0.3, 0.3, p=1.0),
                        A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    ],
                    p=0.5,
                ),
                A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=30, p=0.4),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.MotionBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.25,
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=img_size // 10,
                    max_width=img_size // 10,
                    fill_value=0,
                    p=0.30,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    eval_tf = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_tf, eval_tf
