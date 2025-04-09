from torch.utils.data import Dataset
import numpy as np


class SatelliteDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image and mask from the data
        image = self.data[idx]['image']
        mask = self.data[idx]['mask']

        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.int16)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
