import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class PortDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_path, sep=";")
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Columna 0: nombre de la imagen, columna 1: etiqueta
        img_name = self.data.iloc[idx, 0]
        label = torch.tensor(int(self.data.iloc[idx, 1]), dtype=torch.long)
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, img_name
