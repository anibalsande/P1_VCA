import os
from matplotlib import image
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from dataset import PortDataset
from transforms import base_transform, aug_transform
from split import split_indices
from model import get_resnet18
from train import train_epoch
from evaluate import evaluate

output_dir = '../results'
seed = 42

# Configuración inicial
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def visualize_samples(dataset, num_samples=5):
    plt.figure(figsize=(15, 5))
    indices = random.sample(range(len(dataset)), num_samples)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, idx in enumerate(indices):
        image, label, name = dataset[idx]
        img = image.permute(1, 2, 0).numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)
            
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {label}\n{name[:15]}")
        plt.axis('off')
    plt.show()


def plot_loss(losses, exp_name):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', color='b')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.title(f'Training Loss - {exp_name}')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'loss_{exp_name}.png'))
    plt.close()


def run_experiment(pretrained, augmentation, csv_path, img_dir, task_name=None):
    exp_name = f"{task_name}_pre_{pretrained}_aug_{augmentation}"
    print(f"\nEXPERIMENT: {exp_name} -----------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_train = aug_transform if augmentation else base_transform
    t_test = base_transform # El test nunca lleva augmentation

    df = pd.read_csv(csv_path, sep=";")
    train_idx, test_idx = split_indices(len(df), train_ratio=0.8)

    # Crear subsets
    train_dataset = Subset(PortDataset(img_dir, csv_path, transform=t_train), train_idx)
    test_dataset = Subset(PortDataset(img_dir, csv_path, transform=t_test), test_idx)

    # Visualizamos algunas muestras solo en la primera configuración sin preentrenamiento ni augmentación
    if pretrained is False and augmentation is False:
        print("Previsualizando datos...")
        visualize_samples(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = get_resnet18(num_classes=2, pretrained=pretrained).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01) # AdamW con weight decay para mejor generalización

    epochs = 40
    epoch_losses = []

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        epoch_losses.append(loss)
        
        # Solo dibujamos visualizaciones en la última época
        is_final_epoch = (epoch == epochs - 1)
        
        test_acc = evaluate(model, test_loader, device, exp_name, epoch, plot_visuals=is_final_epoch)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f} | Test Acc: {test_acc:.4f}")

    # Guardar la gráfica de pérdidas
    plot_loss(epoch_losses, exp_name)

def run_all_experiments(csv_path, img_dir, task_name=None):
    configurations = [
        (False, False),
        (True, False),
        (False, True),
        (True, True)
    ]

    for pretrained, augmentation in configurations:
        run_experiment(pretrained, augmentation, csv_path, img_dir, task_name)

def main():
    # TAREA 2: Clasificación de Barcos
    run_all_experiments(
        csv_path="../P1-Material/ship.csv",
        img_dir="../P1-Material/images",
        task_name="ship")

    # TAREA 4: Clasificación de Barcos Atracados
    run_all_experiments(
        csv_path="../P1-Material/docked.csv",
        img_dir="../P1-Material/images",
        task_name="docked")


if __name__ == "__main__":
    main()