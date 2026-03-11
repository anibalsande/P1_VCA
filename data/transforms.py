import torchvision.transforms as t


def get_transforms(data_augmentation=False):
    
    if data_augmentation:

        train_transform = t.Compose([
            t.Resize((224,224)),
            t.RandomHorizontalFlip(),
            t.RandomRotation(10),
            t.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),
            t.ToTensor(),
            t.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    else:

        train_transform = t.Compose([
            t.Resize((224,224)),
            t.ToTensor(),
            t.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    val_transform = t.Compose([
        t.Resize((224,224)),
        t.ToTensor(),
        t.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    return train_transform, val_transform


if __name__ == "__main__":
    print("Testing transforms module")