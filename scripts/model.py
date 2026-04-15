import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes=2, pretrained=False):

    if pretrained:
        model = models.resnet18(weights="IMAGENET1K_V1")
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":

    model = get_resnet18()
    print(model)