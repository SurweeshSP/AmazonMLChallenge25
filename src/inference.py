import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def predict(model, test_loader, config):
    """Generate predictions for test set"""
    model.eval()
    all_preds = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch['pixel_values'].to(config.DEVICE)
            texts = batch['texts']
            numeric_features = batch['numeric_features'].to(config.DEVICE)
            sample_ids = batch['sample_id']
            
            pred = model(images, texts, numeric_features)
            pred = np.expm1(pred.cpu().numpy())
            
            # Clip predictions to reasonable range
            pred = np.clip(pred, 0, 1e6)
            
            all_preds.extend(pred)
            all_sample_ids.extend(sample_ids)
    
    output_df = pd.DataFrame({
        'sample_id': all_sample_ids,
        'price': all_preds
    })
    
    return output_df

def load_best_model(model, config):
    """Load best checkpoint with SWA handling"""
    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    
    # Handle SWA module prefix
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if present (from SWA)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Remove frozen model weights (vision_encoder, text_encoder)
    state_dict = {k: v for k, v in state_dict.items() 
                  if not k.startswith('vision_encoder.') and not k.startswith('text_encoder.')}
    
    # Load with strict=False to ignore frozen encoder weights
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded best model with validation SMAPE: {checkpoint['val_smape']:.2f}%")
    return model
