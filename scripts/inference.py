import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

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

def run_inference(args):
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carga de modelo y datos
    model = load_model(args.model, device)
    dataset = PortDataset(args.img_dir, args.csv, transform=base_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print(f"Evaluando {len(dataset)} imagenes para la tarea: {args.task}")
    
    # Evaluación
    metrics = evaluate(model, loader, device)

    # Resultados por consola
    acc = metrics["accuracy"]
    print(f"\n--- Resultados [{args.task.upper()}] ---")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    report = classification_report(
        metrics["all_labels"],
        metrics["all_preds"],
        target_names=["No", "Si"]
    )
    print("\nReporte de Clasificación:")
    print(report)

    # 5. Generar y guardar gráficos
    tag = f"inf_{args.task}"
    plot_confusion_matrix(metrics["all_labels"], metrics["all_preds"], tag, args.output)
    plot_roc_curve(metrics["all_labels"], metrics["all_probs"], tag, args.output)
    plot_misclassified(metrics["misclassified"], tag, args.output)
    
    print(f"\nGraficos guardados en: {args.output}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--task", default="ship")
    parser.add_argument("--output", default="./results_inference")

    args = parser.parse_args()
    run_inference(args)