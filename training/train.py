import torch


def train_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    print("Train loss:", avg_loss)


if __name__ == "__main__":
    print("Testing train module")