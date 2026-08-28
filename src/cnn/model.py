"""Arquitetura da CNN para mamografias (PyTorch + MobileNetV2)."""
import torch
import torch.nn as nn
import torchvision.models as models


def build_mammography_cnn(
    n_classes: int = 1,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    MobileNetV2 como feature extractor + head customizado.
    
    Args:
        n_classes: Numero de classes (1 para binario sigmoid)
        pretrained: Usar pesos ImageNet
        freeze_backbone: Congelar backbone inicialmente
    """
    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
    else:
        weights = None

    backbone = models.mobilenet_v2(weights=weights)
    
    # Remove o classificador original
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    class MammoCNN(nn.Module):
        def __init__(self, backbone, in_features, n_classes):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)

    return MammoCNN(backbone, in_features, n_classes)


def unfreeze_for_finetune(model: nn.Module, num_unfrozen_layers: int = 20):
    """Descongela as ultimas `num_unfrozen_layers` camadas do backbone."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    layers = list(model.backbone.features)
    for layer in layers[-num_unfrozen_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


def count_parameters(model: nn.Module) -> dict:
    """Conta parametros treinaveis e totais."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
