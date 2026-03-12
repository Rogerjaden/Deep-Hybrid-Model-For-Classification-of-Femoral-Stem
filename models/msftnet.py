import torch
import torch.nn as nn
import timm
from models.eca import ECAModule
from models.cbam import CBAM

class MSFTNet(nn.Module):
    """
    MSFT-Net: Multi-Scale Feature Transformer Network.
    
    A hybrid architecture combining:
    1. ResNet-50 Backbone for local feature extraction.
    2. Configurable Attention (ECA/CBAM) for feature recalibration.
    3. Transformer Encoder for global context modeling.
    """
    def __init__(self, num_classes, attn_type="eca"):
        super().__init__()

        # Feature extraction backbone (ResNet-50)
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=True,
            features_only=True
        )

        # Dimension of C5 feature maps for ResNet50
        in_channels = 2048 
        
        # Attention Module selection
        if attn_type == "eca":
            self.attn = ECAModule(in_channels)
        elif attn_type == "cbam":
            self.attn = CBAM(in_channels)
        else:
            self.attn = nn.Identity()

        # Transformer layer for modeling spatial dependencies as a sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2048,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Final classification head
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        Forward pass logic:
        Image -> CNN Backbone -> Attention -> Reshape to Seq -> Transformer -> GAP -> Classifier
        """
        # Step 1: CNN Feature Extraction - using the final feature map (C5)
        x = self.backbone(x)[-1] # [B, 2048, 7, 7]
        
        # Step 2: Apply Recalibration Attention
        x = self.attn(x)

        b, c, h, w = x.shape

        # Step 3: Transform 2D feature maps into a 1D sequence for the Transformer
        # Seq length N = 7*7 = 49
        x = x.view(b, c, h * w).permute(0, 2, 1)  # [B, 49, 2048]

        # Step 4: Model global context via Transformer Encoder
        x = self.transformer(x)

        # Step 5: Global Average Pooling (GAP) across the sequence dimension
        x = x.mean(dim=1)  # [B, 2048]

        # Step 6: Classification
        return self.fc(x)