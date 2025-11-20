import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np

def smape_loss(pred, target):
    """Symmetric Mean Absolute Percentage Error"""
    numerator = torch.abs(pred - target)
    denominator = (torch.abs(target) + torch.abs(pred)) / 2
    return torch.mean(numerator / (denominator + 1e-8))

def mape_loss(pred, target):
    """Mean Absolute Percentage Error"""
    return torch.mean(torch.abs((target - pred) / (target + 1e-8)))

def calculate_smape(pred, target):
    """Calculate SMAPE metric"""
    pred = np.expm1(pred)
    target = np.expm1(target)
    # Clip predictions to reasonable range
    pred = np.clip(pred, 0, 1e6)
    numerator = np.abs(pred - target)
    denominator = (np.abs(target) + np.abs(pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # SWA model
        if config.USE_SWA:
            self.swa_model = AveragedModel(model)
            self.swa_start = config.NUM_EPOCHS // 2
        
        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        num_training_steps = len(train_loader) * config.NUM_EPOCHS
        num_warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5
        )
        
        if config.USE_SWA:
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=config.MIN_LR)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss(beta=0.5)
        
        self.best_smape = float('inf')
        self.scaler = torch.amp.GradScaler('cuda') if config.USE_AMP else None
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch in pbar:
            images = batch['pixel_values'].to(self.config.DEVICE)
            texts = batch['texts']
            numeric_features = batch['numeric_features'].to(self.config.DEVICE)
            target = batch['price'].to(self.config.DEVICE)
            
            if self.config.USE_AMP:
                with torch.amp.autocast('cuda'):
                    pred = self.model(images, texts, numeric_features)
                    # Multi-objective loss
                    loss = (0.25 * self.mse_loss(pred, target) + 
                           0.25 * self.huber_loss(pred, target) +
                           0.25 * self.mae_loss(pred, target) +
                           0.25 * smape_loss(pred, target))
            else:
                pred = self.model(images, texts, numeric_features)
                loss = (0.25 * self.mse_loss(pred, target) + 
                       0.25 * self.huber_loss(pred, target) +
                       0.25 * self.mae_loss(pred, target) +
                       0.25 * smape_loss(pred, target))
            
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
            
            # Update scheduler
            if self.config.USE_SWA and epoch >= self.swa_start:
                self.swa_scheduler.step()
            else:
                self.scheduler.step()
            
            total_loss += loss.item()
            all_preds.extend(pred.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Update SWA model
        if self.config.USE_SWA and epoch >= self.swa_start:
            self.swa_model.update_parameters(self.model)
        
        avg_loss = total_loss / len(self.train_loader)
        train_smape = calculate_smape(np.array(all_preds), np.array(all_targets))
        
        return avg_loss, train_smape
    
    def validate(self, use_swa=False):
        model = self.swa_model if (use_swa and self.config.USE_SWA) else self.model
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['pixel_values'].to(self.config.DEVICE)
                texts = batch['texts']
                numeric_features = batch['numeric_features'].to(self.config.DEVICE)
                target = batch['price'].to(self.config.DEVICE)
                
                pred = model(images, texts, numeric_features)
                loss = self.mse_loss(pred, target)
                total_loss += loss.item()
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        val_smape = calculate_smape(np.array(all_preds), np.array(all_targets))
        
        return avg_loss, val_smape
    
    def train(self):
        print("Starting training...")
        self.config.CHECKPOINT_DIR.mkdir(exist_ok=True)
        
        patience = 7
        patience_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_smape = self.train_epoch(epoch)
            
            # Validate with SWA model if available
            use_swa = self.config.USE_SWA and epoch >= self.swa_start
            val_loss, val_smape = self.validate(use_swa=use_swa)
            
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}, Train SMAPE: {train_smape:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val SMAPE: {val_smape:.2f}%" + 
                  (" (SWA)" if use_swa else ""))
            
            if val_smape < self.best_smape:
                self.best_smape = val_smape
                patience_counter = 0
                
                # Save the appropriate model
                model_to_save = self.swa_model if use_swa else self.model
                checkpoint_path = self.config.CHECKPOINT_DIR / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict() if use_swa else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_smape': val_smape,
                }, checkpoint_path)
                print(f"✓ Saved best model with SMAPE: {val_smape:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\n✓ Training completed! Best SMAPE: {self.best_smape:.2f}%")
