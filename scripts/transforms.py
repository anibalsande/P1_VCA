import torchvision.transforms as t

normalize = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

base_transform = t.Compose([
    t.Resize((224, 224)),
    t.ToTensor(),
    normalize
])

aug_transform = t.Compose([
    t.Resize((224, 224)),
    t.RandomHorizontalFlip(p=0.5),
    t.RandomRotation(degrees=5),
    t.ColorJitter(brightness=0.2, contrast=0.2),
    t.ToTensor(),
    normalize
])