import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18(nn.Module):
    def __init__(self, freeze_bn, input_channels, feature_size):
        super(ResNet18, self).__init__()
        self.feature_size = feature_size

        self.boundary_encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # 暂时先不考虑freeze_batch_norm的情况

        self.boundary_encoder.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )

        self.boundary_encoder.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.feature_size)
        )
        self.boundary_encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.boundary_encoder(x)

def get_boundary_encoder(
    name,
    freeze_bn=False,
    input_channels=1,
    feature_size=256,
):
    return {
        "resnet18": ResNet18(
            freeze_bn=freeze_bn,
            input_channels=input_channels,
            feature_size=feature_size,
        ),
    }[name]