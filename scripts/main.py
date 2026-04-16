import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt


from dataset import PortDataset
from transforms import base_transform, aug_transform
from split import split_indices
from model import get_resnet18
from train import train_epoch
from evaluate import evaluate
from plots import *

# Configuración general
SEED       = 42
EPOCHS     = 50
BATCH_SIZE = 128
OUTPUT_DIR = '../results'
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def visualize_samples(dataset, num_samples=5):
    """Muestra num_samples imágenes del dataset con sus etiquetas."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
 
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(indices):
        image, label, name = dataset[idx]
        img = image.permute(1, 2, 0).numpy() * std + mean
        img = np.clip(img, 0, 1)
        plt.subplot(1, len(indices), i + 1)
        plt.imshow(img)
        plt.title(f"Label: {label.item()}\n{os.path.basename(name)[:15]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"))
    plt.show()
    print(f"Muestra de imágenes guardada en {OUTPUT_DIR}/sample_images.png")


def save_model(model, exp_name):
    path = os.path.join(MODELS_DIR, f"{exp_name}.pth")
    torch.save(model.state_dict(), path)
    print(f"Modelo guardado en {path}")
    return path


def run_experiment(pretrained, augmentation, train_dataset, test_dataset,
                   task_name, all_results):
    
    exp_name = f"{task_name}_pre_{pretrained}_aug_{augmentation}"
    print(f"\nEXPERIMENTO: {exp_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    model     = get_resnet18(num_classes=2, pretrained=pretrained).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01) # AdamW con weight decay para mejor generalización

    # Bucle de entrenamiento
    epoch_losses = []
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        epoch_losses.append(loss)
        
        metrics  = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss:.4f} | Test Acc: {metrics['accuracy']:.4f}")

    # Evaluación final
    train_metrics = evaluate(model, train_loader, device)
    test_metrics  = evaluate(model, test_loader,  device)

    save_model(model, exp_name)
    all_results[exp_name] = {
        "train_acc":     train_metrics["accuracy"],
        "test_acc":      test_metrics["accuracy"],
        "epoch_losses":  epoch_losses,
        "test_metrics":  test_metrics,
    }


def run_all_experiments(csv_path, img_dir, task_name, all_results,
                        show_samples=False):
    configurations = [
        (False, False),
        (True, False),
        (False, True),
        (True, True)
    ]

    df = pd.read_csv(csv_path, sep=";")
    train_idx, test_idx = split_indices(len(df), train_ratio=0.8, seed=SEED)

    if show_samples:
        raw_dataset = PortDataset(img_dir, csv_path, transform=base_transform)
        visualize_samples(raw_dataset, num_samples=5)

    for pretrained, augmentation in configurations:
        t_train = aug_transform if augmentation else base_transform
 
        train_dataset = Subset(
            PortDataset(img_dir, csv_path, transform=t_train), train_idx)
        test_dataset  = Subset(
            PortDataset(img_dir, csv_path, transform=base_transform), test_idx)
 
        run_experiment(
            pretrained=pretrained,
            augmentation=augmentation,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            task_name=task_name,
            all_results=all_results,
        )
 

def generate_all_plots(all_results, class_names=None):
    print(f"\n{'='*60}")
    print("GENERANDO GRÁFICOS...")
    print(f"{'='*60}")
 
    for exp_name, data in all_results.items():
        print(f"\n  {exp_name}")
        metrics = data["test_metrics"]
 
        plot_loss(data["epoch_losses"], exp_name, OUTPUT_DIR)
        plot_confusion_matrix(
            metrics["all_labels"], metrics["all_preds"],
            exp_name, OUTPUT_DIR, class_names=class_names
        )
        plot_roc_curve(
            metrics["all_labels"], metrics["all_probs"],
            exp_name, OUTPUT_DIR
        )
        plot_misclassified(metrics["misclassified"], exp_name, OUTPUT_DIR)
 
    # Gráfico resumen comparando todos los experimentos
    plot_accuracy_summary(all_results, OUTPUT_DIR)

def main():
    all_results = {}   # Acumula resultados de todos los experimentos
 
    # TAREA 2: Clasificación Ship / No-ship
    run_all_experiments(
        csv_path="../P1-Material/ship.csv",
        img_dir="../P1-Material/images",
        task_name="ship",
        all_results=all_results,
        show_samples=True,          # Solo muestra muestras una vez
    )
 
    # TAREA 4 (opcional): Clasificación Docked / Undocked
    run_all_experiments(
        csv_path="../P1-Material/docked.csv",
        img_dir="../P1-Material/images",
        task_name="docked",
        all_results=all_results,
        show_samples=False,
    )
 
    # Gráficos al final de todo
    generate_all_plots(all_results)
 
    # Resumen por consola
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"{'Experimento':<45} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 67)
    for exp_name, data in all_results.items():
        print(f"{exp_name:<45} {data['train_acc']:>10.4f} {data['test_acc']:>10.4f}")
 
    print(f"\nModelos guardados en: {MODELS_DIR}/")
    print(f"Gráficos guardados en: {OUTPUT_DIR}/")
 
 
if __name__ == "__main__":
    main()
