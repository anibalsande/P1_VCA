"""
inference.py — Evaluación de un modelo sobre un dataset de test externo.

Uso típico (defensa de la práctica):
    python inference.py \\
        --model   results/models/ship_pre1_aug1.pth \\
        --csv     /ruta/al/test.csv \\
        --img_dir /ruta/a/las/imagenes \\
        --task    ship

"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import PortDataset
from transforms import base_transform
from model import get_resnet18
from evaluate import evaluate
from plots import plot_confusion_matrix, plot_roc_curve, plot_misclassified


def load_model(model_path, device, num_classes=2):
    model = get_resnet18(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Modelo cargado desde: {model_path}")
    return model


def run_inference(model_path, csv_path, img_dir, task_name, output_dir, num_classes=2):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device, num_classes=num_classes)

    dataset = PortDataset(img_dir, csv_path, transform=base_transform)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Evaluando {len(dataset)} imágenes:")
    metrics = evaluate(model, loader, device)

    acc = metrics["accuracy"]
    print(f"\nResultados sobre '{task_name}':")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")

    # Métricas por clase
    from sklearn.metrics import classification_report
    report = classification_report(
        metrics["all_labels"],
        metrics["all_preds"],
        target_names=["No", "Sí"],
        digits=4
    )
    print("\n" + report)

    # Guardar gráficos
    exp_name = f"inference_{task_name}"
    plot_confusion_matrix(metrics["all_labels"], metrics["all_preds"],
                          exp_name, output_dir)
    plot_roc_curve(metrics["all_labels"], metrics["all_probs"],
                   exp_name, output_dir)
    plot_misclassified(metrics["misclassified"], exp_name, output_dir)

    print(f"\nGráficos guardados en: {output_dir}/")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evalúa un modelo entrenado sobre un dataset de test externo."
    )
    parser.add_argument("--model",   required=True,
                        help="Ruta al .pth del modelo entrenado")
    parser.add_argument("--csv",     required=True,
                        help="CSV del dataset de test")
    parser.add_argument("--img_dir", required=True,
                        help="Directorio con las imágenes de test")
    parser.add_argument("--task",    required=True,
                        help="Nombre descriptivo (p.ej. ship o docked)")
    parser.add_argument("--output",  default="inference_results",
                        help="Carpeta de salida para los gráficos")

    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        csv_path=args.csv,
        img_dir=args.img_dir,
        task_name=args.task,
        output_dir=args.output,
    )