import torch
from torch.utils.data import DataLoader

from data.dataset import PortDataset
from data.transforms import get_transforms
from utils.split import split_dataset
from models.model import get_resnet18
from training.train import train_epoch
from training.evaluate import evaluate


def run_experiment(pretrained, augmentation):

    print("\n===============================")
    print("Pretrained:", pretrained)
    print("Data Augmentation:", augmentation)
    print("===============================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, val_transform = get_transforms(
        data_augmentation=augmentation
    )

    dataset = PortDataset(
        csv_file="P1-Material/ship.csv",
        img_dir="P1-Material/images",
        label_column="Ship/No-Ship",
        transform=train_transform
    )

    train_dataset, val_dataset = split_dataset(dataset)

    # cambiar transform de validation
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    model = get_resnet18(
        num_classes=2,
        pretrained=pretrained
    )

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    epochs = 10

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_acc = evaluate(
            model,
            val_loader,
            device
        )

        print("Validation accuracy:", val_acc)


def main():

    experiments = [
        {"pretrained": False, "augmentation": False},
        {"pretrained": False, "augmentation": True},
        {"pretrained": True, "augmentation": False},
        {"pretrained": True, "augmentation": True},
    ]

    for exp in experiments:

        run_experiment(
            pretrained=exp["pretrained"],
            augmentation=exp["augmentation"]
        )


if __name__ == "__main__":
    main()