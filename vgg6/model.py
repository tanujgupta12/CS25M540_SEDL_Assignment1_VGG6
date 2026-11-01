"""VGG6 for CIFAR-10 with a configurable activation function."""
import torch.nn as nn

# Four 3x3 conv layers + two fully connected layers = 6 weight layers.
VGG6_CFG = [64, 64, "M", 128, 128, "M"]

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def get_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


class VGG6(nn.Module):
    def __init__(self, activation="relu", num_classes=10, dropout=0.5):
        super().__init__()
        layers, in_ch = [], 3
        for v in VGG6_CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1), nn.BatchNorm2d(v), get_activation(activation)]
                in_ch = v
        self.features = nn.Sequential(*layers)
        # 32x32 input, two max-pools -> 8x8 feature map
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 8 * 8, 256),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
