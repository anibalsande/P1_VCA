import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def evaluate(model, loader, device):
    """
    Evalúa el modelo sobre un DataLoader.

    Devuelve un diccionario con:
        - accuracy
        - all_preds:   lista de predicciones (clase argmax)
        - all_labels:  lista de etiquetas reales
        - all_probs:   lista de probabilidades de la clase positiva (para ROC)
        - misclassified: lista de tuplas (imagen_tensor, pred, label) para visualización
    """
    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []
    misclassified = []   # (img_tensor_cpu, pred, true_label)

    with torch.no_grad():
        for images, labels, _ in loader:
            images_dev = images.to(device)
            labels_dev = labels.to(device)

            outputs = model(images_dev)
            probs = torch.softmax(outputs, dim=1)[:, 1]   # P(clase positiva)
            preds = outputs.argmax(1)

            correct += (preds == labels_dev).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_dev.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Guardar imágenes mal clasificadas (máximo 12)
            if len(misclassified) < 12:
                mismatches = (preds != labels_dev).nonzero(as_tuple=True)[0]
                for idx in mismatches:
                    if len(misclassified) < 12:
                        misclassified.append((
                            images[idx].cpu(),
                            preds[idx].item(),
                            labels_dev[idx].item()
                        ))

    return {
        "accuracy":      correct / total,
        "all_preds":     np.array(all_preds),
        "all_labels":    np.array(all_labels),
        "all_probs":     np.array(all_probs),
        "misclassified": misclassified,
    }