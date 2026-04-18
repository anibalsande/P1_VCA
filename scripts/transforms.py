import torchvision.transforms as t

normalize = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

base_transform = t.Compose([
    t.Resize((224, 224)),
    t.ToTensor(),
    normalize
])

aug_transform = t.Compose([
    t.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    t.RandomHorizontalFlip(p=0.5),
    t.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.2, hue=0.05),
    t.ToTensor(),
    normalize
])