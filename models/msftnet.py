import torch
import torch.nn as nn
import timm
from models.eca import ECA

class MSFTNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN backbone
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=False,
            num_classes=0
        )

        self.eca = ECA(2048)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2048,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        self.feature_map = x

        x = self.eca(x)

        b, c, h, w = x.shape

        x = x.view(b, c, h * w).permute(0, 2, 1)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.fc(x)