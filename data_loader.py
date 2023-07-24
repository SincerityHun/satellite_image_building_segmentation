import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import SatelliteDataset, DataLoader
import torch
import pandas as pd
import random

IMG_HEIGHT = 224
IMG_WIDTH = 224

train_transform = A.Compose(
    [
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Sharpen(alpha=(0.3, 0.5), lightness=(0.5, 1.0), always_apply=True),
        A.OneOf(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
            ],
            p=1,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(),
                A.RandomBrightness(),
                A.RandomBrightnessContrast(),
            ],
            p=1,
        ),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)
val_transform = A.Compose(
    [
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)

train_ratio = 0.8
# Split indices
indices = list(range(len(pd.read_csv("./data/train.csv"))))
random.shuffle(indices)
split_idx = int(len(indices) * train_ratio)

train_indices = indices[:split_idx]
valid_indices = indices[split_idx:]

# Create datasets
train_dataset = SatelliteDataset(
    csv_file="./data/train.csv", transform=train_transform, indices=train_indices
)
valid_dataset = SatelliteDataset(
    csv_file="./data/train.csv", transform=val_transform, indices=valid_indices
)

# Create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=4
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=16, shuffle=False, num_workers=4
)

# 1. Dataset Creation
test_dataset = SatelliteDataset(
    csv_file="./data/test.csv", transform=val_transform, infer=True
)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
