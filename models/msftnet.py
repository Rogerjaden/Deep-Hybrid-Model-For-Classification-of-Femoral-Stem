import torch
import torch.nn as nn
import timm
from models.cbam import CBAM

class MSFTNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN backbone
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=True,
            features_only=True
        )

        self.cbam = CBAM(2048)

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
        x = self.backbone(x)[-1]
        x = self.cbam(x)

        b, c, h, w = x.shape

        x = x.view(b, c, h * w).permute(0, 2, 1)  # [B,49,2048]

        x = self.transformer(x)

        # FIXED LINE
        x = x.mean(dim=1)  # [B,2048]

        return self.fc(x)