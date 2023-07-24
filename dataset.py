from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from utils import rle_decode


class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False, indices=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        # indices: List of indices to load
        self.indices = indices if indices is not None else list(range(len(self.data)))

    def __len__(self):
        if self.infer:
            return len(self.indices)
        else:
            return len(self.indices) * 16  # 16 sub-images for each image

    def __getitem__(self, idx):
         # The index of the original image this patch comes from
        orig_idx = self.indices[idx // (1 if self.infer else 16)]
        img_path = self.data.iloc[orig_idx, 1]
        img_path = img_path.replace("./", "./data/")  # 경로 변경
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.transform:
                image = self.transform(image=image)["image"]
            return image
        else:
            # Get the original index and the index of the sub-image
            sub_idx = idx % 16

            # Compute the coordinates of the sub-image
            i = sub_idx // 4
            j = sub_idx % 4
            image = image[i * 224 : (i + 1) * 224, j * 224 : (j + 1) * 224, :]

            mask_rle = self.data.iloc[orig_idx, 2]
            # The mask should be decoded with the original size of the image
            mask = rle_decode(mask_rle, (1024, 1024))
            mask = mask[i * 224 : (i + 1) * 224, j * 224 : (j + 1) * 224]

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            return image, mask
