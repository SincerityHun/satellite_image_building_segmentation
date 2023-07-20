import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import SatelliteDataset, DataLoader
import torch

transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
# 1. Dataset Creation
dataset = SatelliteDataset(csv_file="./data/train.csv", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
test_dataset = SatelliteDataset(
    csv_file="./data/test.csv", transform=transform, infer=True
)

# 2. Random Split to get Valid dataset
valid_ratio = 0.2
dataset_size = len(dataloader.dataset)
valid_size = int(dataset_size * valid_ratio)
train_size = dataset_size - valid_size
train_dataset, valid_dataset = torch.utils.data.random_split(
    dataloader.dataset, [train_size, valid_size]
)


# 3. Data Loader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=4
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=16, shuffle=False, num_workers=4
)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
