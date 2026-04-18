import torch

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    # Añadimos '_' para ignorar el nombre de la imagen durante el entrenamiento
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss