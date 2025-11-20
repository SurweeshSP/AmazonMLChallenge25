import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
from transformers import CLIPProcessor
import re
from sklearn.preprocessing import RobustScaler

class OptimizedProductDataset(Dataset):
    def __init__(self, df, image_dir, processor, is_train=True, scaler=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.is_train = is_train
        self.scaler = scaler
        
    def extract_advanced_features(self, text):
        """Enhanced feature extraction with more signals"""
        # Extract IPQ
        ipq_match = re.search(r'IPQ[:\s]*(\d+)', text, re.IGNORECASE)
        ipq = int(ipq_match.group(1)) if ipq_match else 1
        
        # Clean text
        text_clean = re.sub(r'https?://\S+', '', text)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        
        # Extract various features
        text_len = min(len(text_clean), 5000) / 5000.0  # Normalize
        word_count = min(len(text_clean.split()), 500) / 500.0
        
        # Brand detection (expanded list)
        brand_keywords = [
            'Nike', 'Adidas', 'Apple', 'Samsung', 'Sony', 'LG', 'Dell', 'HP', 'Lenovo',
            'Canon', 'Nikon', 'Panasonic', 'Philips', 'Bosch', 'Siemens', 'Microsoft',
            'Google', 'Amazon', 'Walmart', 'Target', 'Nestle', 'Unilever', 'Coca-Cola',
            'Pepsi', 'Intel', 'AMD', 'NVIDIA', 'Logitech', 'Asus', 'Acer', 'MSI'
        ]
        has_brand = 1.0 if any(re.search(rf'\b{brand}\b', text_clean, re.IGNORECASE) 
                                for brand in brand_keywords) else 0.0
        
        # Count numbers
        num_numbers = min(len(re.findall(r'\d+', text_clean)), 50) / 50.0
        
        # Size/measurement indicators
        has_size = 1.0 if re.search(
            r'\d+\s*(ml|kg|g|mg|oz|lb|lbs|inch|inches|cm|mm|ft|feet|l|liters?)', 
            text_clean, re.IGNORECASE
        ) else 0.0
        
        # Category indicators (expanded)
        is_electronics = 1.0 if re.search(
            r'\b(battery|charger|cable|wireless|bluetooth|hdmi|usb|adapter|laptop|phone|tablet|computer|monitor|keyboard|mouse)\b', 
            text_clean, re.IGNORECASE
        ) else 0.0
        
        is_food = 1.0 if re.search(
            r'\b(organic|natural|fresh|vitamin|protein|dietary|nutrition|gluten|vegan|vegetarian|food|snack|drink|beverage)\b', 
            text_clean, re.IGNORECASE
        ) else 0.0
        
        is_clothing = 1.0 if re.search(
            r'\b(shirt|pants|dress|jacket|shoes|socks|hat|gloves|fabric|cotton|polyester|size|xl|medium|small|large)\b',
            text_clean, re.IGNORECASE
        ) else 0.0
        
        is_home = 1.0 if re.search(
            r'\b(furniture|table|chair|sofa|bed|kitchen|bathroom|decor|lamp|curtain|rug|pillow)\b',
            text_clean, re.IGNORECASE
        ) else 0.0
        
        # Price-related keywords
        has_premium = 1.0 if re.search(
            r'\b(premium|luxury|professional|pro|plus|deluxe|elite|advanced)\b',
            text_clean, re.IGNORECASE
        ) else 0.0
        
        # Extract potential pack/quantity numbers
        pack_match = re.search(r'(\d+)\s*pack', text_clean, re.IGNORECASE)
        pack_qty = int(pack_match.group(1)) if pack_match else 1
        
        # Truncate text
        text_model = text_clean[:400]
        
        features = {
            'text': text_model,
            'ipq': np.log1p(ipq),  # Log transform for IPQ
            'pack_qty': np.log1p(pack_qty),
            'text_len': text_len,
            'word_count': word_count,
            'has_brand': has_brand,
            'num_numbers': num_numbers,
            'has_size': has_size,
            'is_electronics': is_electronics,
            'is_food': is_food,
            'is_clothing': is_clothing,
            'is_home': is_home,
            'has_premium': has_premium
        }
        
        return features
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['sample_id']
        
        # Load image
        image_path = self.image_dir / f"{sample_id}.jpg"
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='white')
        
        # Process image
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        
        # Extract features
        features = self.extract_advanced_features(row['catalog_content'])
        
        # Create feature vector (13 features now)
        numeric_features = np.array([
            features['ipq'],
            features['pack_qty'],
            features['text_len'],
            features['word_count'],
            features['has_brand'],
            features['num_numbers'],
            features['has_size'],
            features['is_electronics'],
            features['is_food'],
            features['is_clothing'],
            features['is_home'],
            features['has_premium'],
            features['ipq'] * features['pack_qty']  # Interaction term
        ], dtype=np.float32)
        
        result = {
            'pixel_values': pixel_values,
            'text': features['text'],
            'numeric_features': torch.from_numpy(numeric_features),
            'sample_id': sample_id
        }
        
        if self.is_train:
            # More robust price transformation
            price = row['price']
            # Apply box-cox like transformation
            price_transformed = np.log1p(price)
            result['price'] = torch.tensor(price_transformed, dtype=torch.float32)
        
        return result

def collate_fn(batch):
    """Custom collate function"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    texts = [item['text'] for item in batch]
    numeric_features = torch.stack([item['numeric_features'] for item in batch])
    sample_ids = [item['sample_id'] for item in batch]
    
    result = {
        'pixel_values': pixel_values,
        'texts': texts,
        'numeric_features': numeric_features,
        'sample_id': sample_ids
    }
    
    if 'price' in batch[0]:
        prices = torch.stack([item['price'] for item in batch])
        result['price'] = prices
    
    return result

def create_dataloaders(config):
    """Create train and validation dataloaders"""
    from sklearn.model_selection import train_test_split
    
    train_df = pd.read_csv(config.TRAIN_CSV)
    
    # Remove extreme outliers
    price_lower = train_df['price'].quantile(0.001)
    price_upper = train_df['price'].quantile(0.999)
    train_df = train_df[(train_df['price'] >= price_lower) & (train_df['price'] <= price_upper)]
    
    # Stratified split by price bins
    try:
        train_df, val_df = train_test_split(
            train_df, 
            test_size=config.VAL_SPLIT, 
            random_state=config.SEED,
            stratify=pd.qcut(train_df['price'], q=20, duplicates='drop')
        )
    except:
        train_df, val_df = train_test_split(
            train_df, 
            test_size=config.VAL_SPLIT, 
            random_state=config.SEED
        )
    
    processor = CLIPProcessor.from_pretrained(config.VISION_MODEL)
    
    train_dataset = OptimizedProductDataset(train_df, config.IMAGE_DIR, processor, is_train=True)
    val_dataset = OptimizedProductDataset(val_df, config.IMAGE_DIR, processor, is_train=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, processor

def create_test_dataloader(config, processor):
    """Create test dataloader"""
    test_df = pd.read_csv(config.TEST_CSV)
    test_dataset = OptimizedProductDataset(test_df, config.IMAGE_DIR, processor, is_train=False)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    return test_loader, test_df
