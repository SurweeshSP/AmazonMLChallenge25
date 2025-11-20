import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
from transformers import CLIPProcessor
import re

class EnhancedProductDataset(Dataset):
    def __init__(self, df, image_dir, processor, is_train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.is_train = is_train
        
        # Precompute text features for efficiency
        if is_train:
            self._engineer_features()
        
    def _engineer_features(self):
        """Advanced feature engineering"""
        # Extract numeric features from text
        self.df['text_length'] = self.df['catalog_content'].str.len()
        self.df['word_count'] = self.df['catalog_content'].str.split().str.len()
        self.df['has_brand'] = self.df['catalog_content'].str.contains(
            r'\b(Nike|Adidas|Apple|Samsung|Sony|LG|Dell|HP|Lenovo)\b', 
            case=False, 
            regex=True
        ).astype(int)
        
        # Extract numbers from text (potential prices, quantities, sizes)
        self.df['num_numbers'] = self.df['catalog_content'].apply(
            lambda x: len(re.findall(r'\d+', str(x)))
        )
        
    def extract_advanced_features(self, text):
        """Extract comprehensive features from catalog content"""
        # Extract IPQ
        ipq_match = re.search(r'IPQ[:\s]*(\d+)', text, re.IGNORECASE)
        ipq = int(ipq_match.group(1)) if ipq_match else 1
        
        # Clean text
        text_clean = re.sub(r'https?://\S+', '', text)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        
        # Extract features
        text_len = len(text_clean)
        word_count = len(text_clean.split())
        
        # Check for brand mentions
        has_brand = 1 if re.search(
            r'\b(Nike|Adidas|Apple|Samsung|Sony|LG|Dell|HP|Lenovo|Canon|Nikon)\b',
            text_clean,
            re.IGNORECASE
        ) else 0
        
        # Count numbers in text
        num_numbers = len(re.findall(r'\d+', text_clean))
        
        # Extract size-related info
        has_size = 1 if re.search(r'\d+\s*(ml|kg|g|mg|oz|lb|inch|cm|mm)', text_clean, re.IGNORECASE) else 0
        
        # Check for keywords
        is_electronics = 1 if re.search(r'\b(battery|charger|cable|wireless|bluetooth)\b', text_clean, re.IGNORECASE) else 0
        is_food = 1 if re.search(r'\b(organic|natural|fresh|vitamin|protein)\b', text_clean, re.IGNORECASE) else 0
        
        # Truncate text for model
        text_model = text_clean[:400]
        
        features = {
            'text': text_model,
            'ipq': ipq,
            'text_len': text_len,
            'word_count': word_count,
            'has_brand': has_brand,
            'num_numbers': num_numbers,
            'has_size': has_size,
            'is_electronics': is_electronics,
            'is_food': is_food,
            'ipq_log': np.log1p(ipq)
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
        
        # Create feature vector
        numeric_features = torch.tensor([
            features['ipq'],
            features['text_len'] / 1000.0,  # Normalize
            features['word_count'] / 100.0,
            features['has_brand'],
            features['num_numbers'] / 10.0,
            features['has_size'],
            features['is_electronics'],
            features['is_food'],
            features['ipq_log']
        ], dtype=torch.float32)
        
        result = {
            'pixel_values': pixel_values,
            'text': features['text'],
            'numeric_features': numeric_features,
            'sample_id': sample_id
        }
        
        if self.is_train:
            price = np.log1p(row['price'])
            result['price'] = torch.tensor(price, dtype=torch.float32)
        
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
    
    # Remove outliers (prices > 99.9 percentile)
    price_upper = train_df['price'].quantile(0.999)
    train_df = train_df[train_df['price'] <= price_upper]
    
    train_df, val_df = train_test_split(
        train_df, 
        test_size=config.VAL_SPLIT, 
        random_state=config.SEED,
        stratify=pd.qcut(train_df['price'], q=10, duplicates='drop')  # Stratified split
    )
    
    processor = CLIPProcessor.from_pretrained(config.VISION_MODEL)
    
    train_dataset = EnhancedProductDataset(train_df, config.IMAGE_DIR, processor, is_train=True)
    val_dataset = EnhancedProductDataset(val_df, config.IMAGE_DIR, processor, is_train=True)
    
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
    test_dataset = EnhancedProductDataset(test_df, config.IMAGE_DIR, processor, is_train=False)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    return test_loader, test_df
