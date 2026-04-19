import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class PortDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        super().__init__()
        df = pd.read_csv(csv_path, sep=";")
        self.transform = transform
        
        temp_names = df.iloc[:, 0].tolist()
        temp_labels = df.iloc[:, 1].astype(int).tolist()

        # Precarga de datos en RAM
        self.samples = []
        
        for name, label in zip(temp_names, temp_labels):
            img_path = os.path.join(img_dir, name)
            # Carga de la imagen
            image = Image.open(img_path).convert("RGB")
            image.load() 
            
            self.samples.append((
                image, 
                torch.tensor(label, dtype=torch.long), 
                name
            ))

        print(f"Dataset cargado")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label, img_name = self.samples[idx]

        # Aumento de datos
        if self.transform:
            image = self.transform(image)

        return image, label, img_name