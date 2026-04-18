import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes=2, pretrained=False):

    if pretrained:
        model = models.resnet18(weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True 
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Para asegurarnos de que la capa final se entrene
    for param in model.fc.parameters():
        param.requires_grad = True

    return model