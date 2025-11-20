import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.data_loader_optimized import create_dataloaders, create_test_dataloader
from src.model_optimized import OptimizedPricePredictor
from src.train import Trainer
from src.inference import predict, load_best_model
from src.utils import set_seed, count_parameters

def main():
    config = Config()
    set_seed(config.SEED)
    
    print(f"Device: {config.DEVICE}")
    print("Creating optimized dataloaders...")
    train_loader, val_loader, processor = create_dataloaders(config)
    
    print("Initializing optimized model...")
    model = OptimizedPricePredictor(config)
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    print("\nTraining model...")
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    
    print("\nGenerating predictions...")
    model = load_best_model(model, config)
    test_loader, test_df = create_test_dataloader(config, processor)
    predictions = predict(model, test_loader, config)
    
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = config.OUTPUT_DIR / "test_out.csv"
    predictions.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
