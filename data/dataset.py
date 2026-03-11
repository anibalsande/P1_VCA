import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PortDataset(Dataset):

    def __init__(self, csv_file, img_dir, label_column, transform=None):

        self.data = pd.read_csv(csv_file, sep=";")
        self.img_dir = img_dir
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx]["Imagen"]
        label = self.data.iloc[idx][self.label_column]

        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


if __name__ == "__main__":

    print("Testing dataset module...")