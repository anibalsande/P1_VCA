import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class PortDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        super().__init__()
        # 1. Leemos el CSV y extraemos todo lo necesario de una vez
        df = pd.read_csv(csv_path, sep=";")
        self.transform = transform
        
        # Extraemos nombres y etiquetas a listas nativas (mucho más rápidas que Pandas)
        temp_names = df.iloc[:, 0].tolist()
        temp_labels = df.iloc[:, 1].astype(int).tolist()

        # 2. Precarga de datos en RAM
        self.samples = []
        print(f"🚀 Iniciando precarga de {len(df)} muestras en RAM...")
        
        for name, label in zip(temp_names, temp_labels):
            img_path = os.path.join(img_dir, name)
            try:
                # Cargamos la imagen
                image = Image.open(img_path).convert("RGB")
                image.load()  # Asegura que los píxeles estén en RAM
                
                # Guardamos la tupla (Imagen, Etiqueta_Tensor, Nombre)
                # Al guardarlo así, la correspondencia es indestructible
                self.samples.append((
                    image, 
                    torch.tensor(label, dtype=torch.long), 
                    name
                ))
            except Exception as e:
                print(f"⚠️ Error cargando {name}: {e}. Saltando...")

        print(f"✅ Dataset listo: {len(self.samples)} imágenes en RAM.")

    def __len__(self):
        # Usamos la longitud de lo que realmente se cargó con éxito
        return len(self.samples)

    def __getitem__(self, idx):
        # 3. Acceso ultra-rápido: desempaquetado de tupla en RAM
        image, label, img_name = self.samples[idx]

        # La CPU solo trabaja aplicando el aumento de datos
        if self.transform:
            image = self.transform(image)

        return image, label, img_name