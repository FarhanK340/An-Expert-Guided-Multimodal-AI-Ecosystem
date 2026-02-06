"""
Train a single modality expert for MoME+ architecture.

This script trains one ModalityExpert at a time, allowing:
- Larger batch sizes (8+ instead of 2)
- Larger crops (128³ instead of 64³)
- Full specialization per modality

Usage:
    python -m src.training.train_single_expert --modality T1 --epochs 100
    python -m src.training.train_single_expert --modality T1ce --epochs 100
    python -m src.training.train_single_expert --modality T2 --epochs 100
    python -m src.training.train_single_expert --modality FLAIR --epochs 100
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
from typing import Optional

from src.models.mome_expert import ModalityExpert
from src.training.loss_functions import DiceLoss
from src.training.metrics import SegmentationMetrics
from src.utils.logger import get_logger

# Enable cudnn benchmark for faster convolutions (finds optimal algorithm)
torch.backends.cudnn.benchmark = True



logger = get_logger(__name__)


class SingleModalityDataset(Dataset):
    """Dataset that returns a single modality from the 4-channel preprocessed data.
    
    Optimized for speed with:
    - Lazy HDF5 loading (keep file open)
    - Contiguous tensors for fast GPU transfer
    - Simple numpy-based augmentation (no MONAI overhead)
    """
    
    # Modality index mapping
    MODALITY_INDEX = {
        "T1": 0,
        "T1ce": 1,
        "T2": 2,
        "FLAIR": 3
    }
    
    def __init__(self, h5_path: str, modality: str, augment: bool = False, max_crops: Optional[int] = None):
        """
        Args:
            h5_path: Path to HDF5 file with preprocessed crops
            modality: One of "T1", "T1ce", "T2", "FLAIR"
            augment: Whether to apply random augmentation
            max_crops: Optional limit on number of crops to use
        """
        self.h5_path = h5_path
        self.modality = modality
        self.modality_idx = self.MODALITY_INDEX[modality]
        self.augment = augment
        
        # Get metadata and detect file format
        with h5py.File(h5_path, 'r') as f:
            self.num_crops = f.attrs.get("num_crops", len([k for k in f.keys() if k.startswith('crop')]))
            
            # Detect if file is per-modality (new) or combined (old) format
            # New format: image shape is (1, D, H, W)
            # Old format: image shape is (4, D, H, W)
            first_crop = f[f"crop_{0:06d}"]
            self.is_single_modality = (first_crop['image'].shape[0] == 1)
        
        if max_crops and max_crops < self.num_crops:
            self.num_crops = max_crops
        
        # Lazy HDF5 file handle
        self.h5_file = None
        
        format_str = "single-modality" if self.is_single_modality else "combined"
        logger.info(f"Loaded {self.num_crops} crops for {modality} from {h5_path} ({format_str} format)")
    
    def __len__(self):
        return self.num_crops
    
    def _open_h5(self):
        """Lazy loading of HDF5 file - keep open for fast access."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
    
    def __getitem__(self, idx):
        self._open_h5()
        
        # Load from HDF5 (file stays open)
        grp = self.h5_file[f"crop_{idx:06d}"]
        
        # Handle both file formats
        if self.is_single_modality:
            # New format: already single channel (1, D, H, W)
            image = grp['image'][:]
        else:
            # Old format: slice to get single channel from (4, D, H, W)
            image = grp['image'][self.modality_idx:self.modality_idx+1]
        
        mask = grp['mask'][:]  # (3, D, H, W)
        
        # Simple numpy augmentation (much faster than MONAI)
        if self.augment:
            # Random flips
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=3).copy()
                mask = np.flip(mask, axis=3).copy()
        
        # Return contiguous tensors for fast GPU transfer
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).float()
        }
    
    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if self.h5_file is not None:
            self.h5_file.close()


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, scaler):
    """Train for one epoch with AMP + GradScaler."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Non-blocking GPU transfer for speed
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with AMP
        with torch.amp.autocast('cuda'):
            outputs, _ = model(images)
            loss = loss_fn(outputs, masks)
        
        # Backward with GradScaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, loss_fn, metrics, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            with torch.cuda.amp.autocast():
                outputs, _ = model(images)  # Returns (segmentation, features)
                loss = loss_fn(outputs, masks)
            
            total_loss += loss.item()
            
            # Compute metrics
            batch_metrics = metrics.compute_metrics(
                {"segmentation": outputs}, 
                {"mask": masks}
            )
            all_metrics.append(batch_metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key]) and m[key] != float('inf')]
        avg_metrics[key] = np.mean(values) if values else 0.0
    avg_metrics['loss'] = total_loss / len(dataloader)
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train a single modality expert")
    parser.add_argument("--modality", type=str, required=True, 
                        choices=["T1", "T1ce", "T2", "FLAIR"],
                        help="Modality to train")
    parser.add_argument("--train_data", type=str, default=None,
                        help="Training HDF5 path (default: auto-detect per-modality file)")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Validation HDF5 path (default: auto-detect per-modality file)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Auto-construct paths if not specified
    # Prefer per-modality files (new format), fall back to combined (old format)
    data_dir = Path("../data/preprocessed")
    if args.train_data is None:
        per_modality_train = data_dir / f"brats2024_gli_{args.modality}_train.h5"
        combined_train = data_dir / "brats2024_gli_train.h5"
        args.train_data = str(per_modality_train if per_modality_train.exists() else combined_train)
    
    if args.val_data is None:
        per_modality_val = data_dir / f"brats2024_gli_{args.modality}_val.h5"
        combined_val = data_dir / "brats2024_gli_val.h5"
        args.val_data = str(per_modality_val if per_modality_val.exists() else combined_val)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {args.modality} expert on {device}")
    
    # Create datasets (augment=True for training)
    train_dataset = SingleModalityDataset(args.train_data, args.modality, augment=True)
    val_dataset = SingleModalityDataset(args.val_data, args.modality, augment=False)
    
    # num_workers=0 on Windows for HDF5, pin_memory=False (REQUIRED for HDF5!)
    # Legacy config explicitly sets pin_memory=false - it pins entire HDF5 file causing slowdown
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model - single expert
    model = ModalityExpert(
        modality=args.modality,  # Required: modality name
        in_channels=1,           # Single modality
        num_classes=3,           # WT, TC, ET
        base_channels=32,
        depth=4
    ).to(device)
    
    # NOTE: torch.compile() disabled - requires Triton which is Linux-only
    # On Linux, uncomment this for 10-30% speedup:
    # model = torch.compile(model)
    # logger.info("Model compiled with torch.compile()")
    
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    loss_fn = DiceLoss()
    metrics = SegmentationMetrics(num_classes=3, class_names=["WT", "TC", "ET"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Checkpoint directory
    checkpoint_dir = Path("experiments/checkpoints/experts")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Training loop with early stopping
    best_dice = 0.0
    start_epoch = 1
    epochs_without_improvement = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training state
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0.0)
            
            # Check if we are already done
            if start_epoch > args.epochs:
                logger.info(f"Checkpoint is already at epoch {checkpoint['epoch']}, which is >= target epochs {args.epochs}")
                return
                
            logger.info(f"Resuming from epoch {start_epoch} (Best Dice: {best_dice:.4f})")
        else:
            logger.warning(f"Checkpoint file not found: {args.resume}. Starting from scratch.")
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, scaler)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate(model, val_loader, loss_fn, metrics, device)
            
            # Compute mean Dice across classes
            mean_dice = (val_metrics.get('dice_WT', 0) + 
                        val_metrics.get('dice_TC', 0) + 
                        val_metrics.get('dice_ET', 0)) / 3
            
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"WT: {val_metrics.get('dice_WT', 0):.4f}, "
                       f"TC: {val_metrics.get('dice_TC', 0):.4f}, "
                       f"ET: {val_metrics.get('dice_ET', 0):.4f}")
            logger.info(f"Mean Dice: {mean_dice:.4f}")
            
            scheduler.step(mean_dice)
            
            # Save best model
            if mean_dice > best_dice:
                best_dice = mean_dice
                epochs_without_improvement = 0
                
                checkpoint_path = checkpoint_dir / f"expert_{args.modality}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'modality': args.modality,
                    'val_metrics': val_metrics
                }, checkpoint_path)
                logger.info(f"✅ Saved best model (Dice: {best_dice:.4f})")
            else:
                epochs_without_improvement += 5  # Since we validate every 5 epochs
                
            # Early stopping
            if epochs_without_improvement >= args.patience:
                logger.info(f"Early stopping after {epochs_without_improvement} epochs without improvement")
                break
        
        # Save 'last' checkpoint every epoch
        last_checkpoint_path = checkpoint_dir / f"expert_{args.modality}_last.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'modality': args.modality,
            # We don't save val_metrics here as they might not be computed this epoch
        }, last_checkpoint_path)
    
    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Best Dice: {best_dice:.4f}")
    logger.info(f"Checkpoint: {checkpoint_dir / f'expert_{args.modality}_best.pth'}")


if __name__ == "__main__":
    main()
