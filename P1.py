import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as t


class PortDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): ruta al archivo CSV con las etiquetas
            img_dir (str): directorio donde están las imágenes
            transform (callable, optional): transformaciones a aplicar
        """

        self.data = pd.read_csv(csv_file, sep=";")
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """Número total de muestras del dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Carga una imagen y su etiqueta"""

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx]["Imagen"]
        label = self.data.iloc[idx]["Ship/No-Ship"]

        img_path = os.path.join(self.img_dir, img_name)

        # Las imágenes ya están en RGB
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
    

transform = t.Compose([
    t.Resize((224, 224)),
    t.ToTensor(),
    t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


dataset = PortDataset(
    csv_file="P1-Material\\ship.csv",
    img_dir="P1-Material\\images",
    transform=transform
)


loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

if __name__ == "__main__":

    print(f"Total samples: {len(dataset)}")
    print(f"Total batches: {len(loader)}")

    for images, labels in loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break