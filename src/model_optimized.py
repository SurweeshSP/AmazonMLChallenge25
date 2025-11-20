import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

class AttentiveFeatureTransformer(nn.Module):
    """Feature transformer with attention mechanism"""
    def __init__(self, input_dim, output_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim)
        )
        self.norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # Add sequence dimension
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        x = self.norm1(x + attn_out.squeeze(1))
        out = self.ff(x)
        return self.norm2(out)

class OptimizedPricePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Frozen encoders
        self.vision_encoder = CLIPVisionModel.from_pretrained(config.VISION_MODEL)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_encoder.eval()
        
        self.text_encoder = SentenceTransformer(config.TEXT_MODEL)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        
        # Attentive feature transformers
        self.vision_transform = AttentiveFeatureTransformer(768, 384, num_heads=8)
        self.text_transform = AttentiveFeatureTransformer(384, 384, num_heads=6)
        
        # Numeric features (13 features now) - FIXED
        self.numeric_transform1 = nn.Sequential(
            nn.Linear(13, 128),  # CHANGED from 9 to 13
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.numeric_transform2 = nn.Sequential(
            nn.Linear(13, 64),  # CHANGED from 9 to 13
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(384, 8, batch_first=True)
        
        # Feature importance gates
        self.vision_gate = nn.Sequential(
            nn.Linear(384, 384),
            nn.Sigmoid()
        )
        
        self.text_gate = nn.Sequential(
            nn.Linear(384, 384),
            nn.Sigmoid()
        )
        
        # Deep residual regression head
        self.regressor = nn.Sequential(
            nn.Linear(384 * 2 + 128 + 64, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, images, texts, numeric_features):
        batch_size = images.size(0)
        
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
        
        # Cross-modal attention
        vision_seq = vision_transformed.unsqueeze(1)
        text_seq = text_transformed.unsqueeze(1)
        cross_attn_out, _ = self.cross_attention(vision_seq, text_seq, text_seq)
        vision_attended = cross_attn_out.squeeze(1)
        
        # Apply feature gates
        vision_gated = vision_attended * self.vision_gate(vision_attended)
        text_gated = text_transformed * self.text_gate(text_transformed)
        
        # Numeric features (dual pathway)
        numeric_feat1 = self.numeric_transform1(numeric_features)
        numeric_feat2 = self.numeric_transform2(numeric_features)
        
        # Concatenate all features
        fused = torch.cat([vision_gated, text_gated, numeric_feat1, numeric_feat2], dim=-1)
        
        # Predict
        output = self.regressor(fused)
        return output.squeeze(-1)
