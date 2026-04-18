"""
plots.py — Funciones de visualización.
Se llaman al final de todos los experimentos, no durante el entrenamiento.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torchvision.transforms as t


# Transformación inversa para visualizar imágenes normalizadas con ImageNet stats
_inv_normalize = t.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def plot_loss(losses, exp_name, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', color='steelblue')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (escala log)')
    plt.title(f'Training Loss — {exp_name}')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    path = os.path.join(output_dir, f'loss_{exp_name}.png')
    plt.savefig(path)
    plt.close()
    print(f"Pérdidas guardadas en {path}")


def plot_confusion_matrix(all_labels, all_preds, exp_name, output_dir, class_names=None):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names or ["0", "1"],
        yticklabels=class_names or ["0", "1"]
    )
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title(f'Confusion Matrix — {exp_name}')
    plt.tight_layout()
    path = os.path.join(output_dir, f'cm_{exp_name}.png')
    plt.savefig(path)
    plt.close()
    print(f"Matriz de confusión guardada en {path}")


def plot_roc_curve(all_labels, all_probs, exp_name, output_dir):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {exp_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, f'roc_{exp_name}.png')
    plt.savefig(path)
    plt.close()
    print(f"Curva ROC guardada en {path} (AUC={roc_auc:.3f})")
    return roc_auc


def plot_misclassified(misclassified, exp_name, output_dir):
    """
    misclassified: lista de tuplas (img_tensor_cpu, pred, true_label)
    """
    if not misclassified:
        print(f"  [plot] Sin errores que mostrar para {exp_name}")
        return

    num_imgs = min(len(misclassified), 12)
    cols = 4
    rows = (num_imgs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

    # Normalizar axes a array plano independientemente de la forma
    axes = np.array(axes).reshape(-1)

    for i in range(num_imgs):
        img_tensor, pred, true_label = misclassified[i]
        img = _inv_normalize(img_tensor).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {pred} | Real: {true_label}", fontsize=9)
        axes[i].axis('off')

    for j in range(num_imgs, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Errores de clasificación — {exp_name}', fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, f'errors_{exp_name}.png')
    plt.savefig(path)
    plt.close()
    print(f"Errores guardados en {path}")


def plot_accuracy_summary(results, output_dir, task_name="general"):
    names = list(results.keys())
    train_accs = [results[n].get("train_acc", 0) for n in names]
    test_accs  = [results[n].get("test_acc",  0) for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
    bars1 = ax.bar(x - width / 2, train_accs, width, label='Train', color='steelblue')
    bars2 = ax.bar(x + width / 2, test_accs,  width, label='Test',  color='coral')

    ax.set_xlabel('Experimento')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Resumen de accuracy por experimento ({task_name.upper()})')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Añadir valor encima de cada barra
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)

    plt.tight_layout()
    filename = f'accuracy_summary_{task_name}.png'
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"\nResumen de accuracy guardado en {path}")