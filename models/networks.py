import torch
import torch.nn as nn
import torchvision.models as models
from .blocks import SEBlock, SPPBlock

class ModifiedResNet101(nn.Module):
    def __init__(self):
        super(ModifiedResNet101, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(resnet101.children())[:-2])
        for name, module in self.features.named_children():
            if 'layer' in name:
                module.add_module("se_block", SEBlock(module[-1].conv2.out_channels))

    def forward(self, x):
        return self.features(x)

class ModifiedEfficientNet(nn.Module):
    def __init__(self):
        super(ModifiedEfficientNet, self).__init__()
        efficient_net = models.efficientnet_b0(pretrained=True)
        self.features = nn.Sequential(*list(efficient_net.children())[:-2])
        self.features.add_module("spp_block", SPPBlock())

    def forward(self, x):
        return self.features(x)

class FusionAndClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(FusionAndClassifier, self).__init__()
        self.resnet101 = ModifiedResNet101()
        self.efficient_net = ModifiedEfficientNet()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self._calculate_input_features(), num_classes)
        )

    def _calculate_input_features(self):
        return 2048 + 1280  # Example for ResNet101 + EfficientNetB0

    def forward(self, x):
        resnet_features = self.resnet101(x)
        efficient_net_features = self.efficient_net(x)
        fused_features = torch.cat([resnet_features, efficient_net_features], dim=1)
        return self.classifier(fused_features)

class HistoCRCNet(nn.Module):
    def __init__(self, num_classes=9):
        super(HistoCRCNet, self).__init__()
        self.resnet101 = ModifiedResNet101()
        self.efficient_net = ModifiedEfficientNet()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self._calculate_input_features(), num_classes)
        )

    def _calculate_input_features(self):
        return 7168  # Example for ResNet101 + EfficientNetB0 with SPP's additional features

    def forward(self, x):
        resnet_features = self.resnet101(x)
        efficient_net_features = self.efficient_net(x)
        fused_features = torch.cat([resnet_features, efficient_net_features], dim=1)
        return self.classifier(fused_features)
