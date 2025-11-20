import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.norm(x + self.net(x))

class ImprovedPricePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Vision encoder (frozen)
        self.vision_encoder = CLIPVisionModel.from_pretrained(config.VISION_MODEL)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_encoder.eval()
        
        # Text encoder (frozen)
        self.text_encoder = SentenceTransformer(config.TEXT_MODEL)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        
        # Feature projections with residual connections
        self.vision_transform = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            ResidualBlock(512, 0.2),
            ResidualBlock(512, 0.2)
        )
        
        self.text_transform = nn.Sequential(
            nn.Linear(384, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            ResidualBlock(512, 0.2),
            ResidualBlock(512, 0.2)
        )
        
        # Numeric features (9 features)
        self.numeric_transform = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256)
        )
        
        # Multimodal fusion with gating
        self.fusion_gate = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.Sigmoid()
        )
        
        # Deep regression head with multiple residual blocks
        self.regressor = nn.Sequential(
            nn.Linear(512 * 2 + 256, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            ResidualBlock(1024, 0.3),
            ResidualBlock(1024, 0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, images, texts, numeric_features):
        # Extract frozen features
        with torch.no_grad():
            vision_feat = self.vision_encoder(pixel_values=images).pooler_output
            text_feat = self.text_encoder.encode(
                texts, convert_to_tensor=True, 
                show_progress_bar=False, device=images.device
            )
        
        # Transform features
        vision_transformed = self.vision_transform(vision_feat.clone().detach())
        text_transformed = self.text_transform(text_feat.clone().detach())
        
        # Gated fusion
        combined = torch.cat([vision_transformed, text_transformed], dim=-1)
        gate = self.fusion_gate(combined)
        vision_gated = vision_transformed * gate
        text_gated = text_transformed * (1 - gate)
        
        # Numeric features
        numeric_transformed = self.numeric_transform(numeric_features)
        
        # Final fusion
        fused = torch.cat([vision_gated, text_gated, numeric_transformed], dim=-1)
        
        # Predict
        output = self.regressor(fused)
        return output.squeeze(-1)
