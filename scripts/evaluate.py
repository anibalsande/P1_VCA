import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchvision.transforms as t

def evaluate(model, loader, device, exp_name, epoch, plot_visuals=False):
    model.eval()
    
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    misclassified_imgs = []
    misclassified_preds = []
    misclassified_labels = []

    # Transformación inversa para visualizar imágenes normalizadas
    inv_normalize = t.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    with torch.no_grad():
        for images, labels, names in loader:
            images_dev = images.to(device)
            labels_dev = labels.to(device)

            outputs = model(images_dev)
            preds = outputs.argmax(1)

            correct += (preds == labels_dev).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_dev.cpu().numpy())

            # Buscar fallos si nos toca graficar
            if plot_visuals:
                mismatches = preds != labels_dev
                if mismatches.any():
                    mismatch_idx = mismatches.nonzero(as_tuple=True)[0]
                    for idx in mismatch_idx:
                        if len(misclassified_imgs) < 12: # Guardamos un máximo de 12 para la cuadrícula
                            img_cpu = inv_normalize(images[idx]).cpu()
                            img_cpu = torch.clamp(img_cpu, 0, 1) # Asegurar rango [0,1] para matplotlib
                            misclassified_imgs.append(img_cpu)
                            misclassified_preds.append(preds[idx].item())
                            misclassified_labels.append(labels_dev[idx].item())

    accuracy = correct / total

    # Generar gráficos solo cuando se solicita (ej. última época)
    if plot_visuals:
        # Matriz de Confusión
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.title(f'Confusion Matrix - {exp_name}')
        plt.tight_layout()
        plt.savefig(f'cm_{exp_name}.png')
        plt.close()

        # Imágenes mal clasificadas
        if misclassified_imgs:
            num_imgs = min(len(misclassified_imgs), 12)
            cols = 4
            rows = (num_imgs + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

            import numpy as np
            if num_imgs == 1:
                axes = np.array([axes]) # Forzar array si solo hay uno
            else:
                axes = axes.flatten() # Convertir matriz a lista plana            


            for i in range(num_imgs):
                img = misclassified_imgs[i].permute(1, 2, 0).numpy()
                axes[i].imshow(img)
                axes[i].set_title(f"Pred: {misclassified_preds[i]} | True: {misclassified_labels[i]}")
                axes[i].axis('off')
                
            for j in range(num_imgs, len(axes)):
                axes[j].axis('off')
                
            plt.tight_layout()
            plt.savefig(f'errors_{exp_name}.png')
            plt.close()

    return accuracy