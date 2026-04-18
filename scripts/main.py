import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt


from dataset import PortDataset
from transforms import base_transform, aug_transform
from split import get_stratified_indexes
from model import get_resnet18

from train import train_epoch
from evaluate import evaluate
from plots import *

# Configuración general
SEED       = 42
EPOCHS     = 50
BATCH_SIZE = 32
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01) # AdamW con weight decay para mejor generalización
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # Historial de métricas por epoch
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    # Bucle de entrenamiento
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluar en Train y Test cada epoch para monitorear progreso
        train_metrics = evaluate(model, train_loader, device)
        test_metrics  = evaluate(model, test_loader, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['test_acc'].append(test_metrics['accuracy'])

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | Test Acc: {test_metrics['accuracy']:.4f}")

        scheduler.step()
    # Guardado
    save_model(model, exp_name)

    print(f"Generando gráficos para {exp_name}...")
    plot_loss(history['train_loss'], exp_name, OUTPUT_DIR) # Puedes mejorar esto luego
    plot_confusion_matrix(test_metrics["all_labels"], test_metrics["all_preds"], exp_name, OUTPUT_DIR)
    plot_roc_curve(test_metrics["all_labels"], test_metrics["all_probs"], exp_name, OUTPUT_DIR)
    plot_misclassified(test_metrics["misclassified"], exp_name, OUTPUT_DIR)

    all_results[exp_name] = {
        "train_acc": train_metrics["accuracy"],
        "test_acc":  test_metrics["accuracy"]
    }

def run_all_experiments(csv_path, img_dir, task_name, all_results,
                        show_samples=False):
    configurations = [
        (False, False),
        (False, True),
        (True, False),
        (True, True)
    ]

    train_idx, test_idx = get_stratified_indexes(csv_path, test_size=0.2, seed=SEED)
    
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

def main():
    print(f"\nTAREA 2: Clasificación Ship / No-ship")
    ship_results = {}
 
    run_all_experiments(
        csv_path="../P1-Material/ship.csv",
        img_dir="../P1-Material/images",
        task_name="ship",
        all_results=ship_results,
        show_samples=True,          # Solo muestra muestras una vez
    )
 
    # Gráfico resumen para los experimentos de Ship
    plot_accuracy_summary(ship_results, OUTPUT_DIR, task_name="ship")

    # TAREA 4 (opcional): Clasificación Docked / Undocked
    print(f"\nTAREA 4: Clasificación Docked / Undocked")
    docked_results = {}
    run_all_experiments(
        csv_path="../P1-Material/docked.csv",
        img_dir="../P1-Material/images",
        task_name="docked",
        all_results=docked_results,
        show_samples=False,
    )
    
    plot_accuracy_summary(docked_results, OUTPUT_DIR, task_name="docked")

    # Resumen final por consola separado
    print("\nRESUMEN FINAL - SHIP")
    for exp, data in ship_results.items():
        print(f"{exp:<35} Train: {data['train_acc']:.4f} | Test: {data['test_acc']:.4f}")
        
    print("\nRESUMEN FINAL - DOCKED")
    for exp, data in docked_results.items():
        print(f"{exp:<35} Train: {data['train_acc']:.4f} | Test: {data['test_acc']:.4f}")

    print(f"\nModelos guardados en: {MODELS_DIR}/")
    print(f"Gráficos guardados en: {OUTPUT_DIR}/") 
 
if __name__ == "__main__":
    main()
