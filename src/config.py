import torch
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    TRAIN_CSV = BASE_DIR / "dataset" / "train.csv"
    TEST_CSV = BASE_DIR / "dataset" / "test.csv"
    IMAGE_DIR = BASE_DIR / "dataset" / "images"
    OUTPUT_DIR = BASE_DIR / "outputs"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    
    # Model settings
    VISION_MODEL = "openai/clip-vit-base-patch32"
    TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    IMAGE_SIZE = 224
    MAX_LENGTH = 128
    
    # Training settings - HEAVILY OPTIMIZED
    BATCH_SIZE = 128 if torch.cuda.is_available() else 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 5e-5
    MIN_LR = 1e-7
    WEIGHT_DECAY = 0.05
    WARMUP_RATIO = 0.05
    
    # Model architecture
    VISION_DIM = 768
    TEXT_DIM = 384
    HIDDEN_DIM = 384
    DROPOUT = 0.3
    NUM_ATTENTION_HEADS = 8
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    
    # Mixed precision
    USE_AMP = torch.cuda.is_available()
    
    # Training tricks
    USE_SWA = True
    LABEL_SMOOTHING = 0.0
    
    # Data split
    VAL_SPLIT = 0.10
    SEED = 42
