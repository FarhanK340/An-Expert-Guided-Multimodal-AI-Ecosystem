"""
Train the fusion network with pre-trained frozen experts.

This script:
1. Loads 4 pre-trained modality experts
2. Freezes all expert weights
3. Trains only the fusion/gating network to combine expert outputs

Usage:
    python -m src.training.train_fusion \
        --expert_t1 experiments/checkpoints/experts/expert_T1_best.pth \
        --expert_t1ce experiments/checkpoints/experts/expert_T1ce_best.pth \
        --expert_t2 experiments/checkpoints/experts/expert_T2_best.pth \
        --expert_flair experiments/checkpoints/experts/expert_FLAIR_best.pth \
        --epochs 50 --batch_size 4
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
from typing import Dict, Optional

from src.models.mome_expert import ModalityExpert
from src.models.mome_segmenter import MoMESegmenter
from src.training.loss_functions import DiceLoss
from src.training.metrics import SegmentationMetrics
from src.utils.logger import get_logger

# MONAI transforms for data augmentation
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d
)

logger = get_logger(__name__)


class MultiModalityDataset(Dataset):
    """Dataset that returns all 4 modalities for fusion training."""
    
    def __init__(self, h5_path: str, transform=None, max_crops: Optional[int] = None):
        """
        Args:
            h5_path: Path to HDF5 file with preprocessed crops
            transform: Optional MONAI transforms
            max_crops: Optional limit on number of crops to use
        """
        self.h5_path = h5_path
        self.transform = transform
        
        # Get crop keys
        with h5py.File(h5_path, 'r') as f:
            self.crop_keys = sorted([k for k in f.keys() if k.startswith('crop')])
        
        if max_crops:
            self.crop_keys = self.crop_keys[:max_crops]
        
        logger.info(f"Loaded {len(self.crop_keys)} crops from {h5_path}")
    
    def __len__(self):
        return len(self.crop_keys)
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[self.crop_keys[idx]]
            image = grp['image'][:]  # (4, D, H, W)
            mask = grp['mask'][:]    # (3, D, H, W)
        
        # Apply transforms if provided
        if self.transform:
            data = {"image": image, "mask": mask}
            data = self.transform(data)
            image, mask = data["image"], data["mask"]
        
        # Split into individual modalities for the model
        return {
            "T1": torch.from_numpy(image[0:1]).float(),
            "T1ce": torch.from_numpy(image[1:2]).float(),
            "T2": torch.from_numpy(image[2:3]).float(),
            "FLAIR": torch.from_numpy(image[3:4]).float(),
            "mask": torch.from_numpy(mask).float()
        }


def load_expert(checkpoint_path: str, device: torch.device) -> ModalityExpert:
    """Load a pre-trained expert from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    expert = ModalityExpert(
        in_channels=1,
        num_classes=3,
        base_channels=32,
        depth=4
    )
    expert.load_state_dict(checkpoint['model_state_dict'])
    expert.to(device)
    
    modality = checkpoint.get('modality', 'Unknown')
    best_dice = checkpoint.get('best_dice', 0)
    logger.info(f"Loaded {modality} expert (Dice: {best_dice:.4f})")
    
    return expert


def create_mome_from_experts(experts: Dict[str, ModalityExpert], device: torch.device) -> MoMESegmenter:
    """Create MoMESegmenter and load pre-trained expert weights."""
    
    # Create MoME model
    model = MoMESegmenter(
        modalities=["T1", "T1ce", "T2", "FLAIR"],
        in_channels=1,
        num_classes=3,
        base_channels=32,
        depth=4
    ).to(device)
    
    # Copy expert weights
    for modality, expert in experts.items():
        model.experts[modality].load_state_dict(expert.state_dict())
        logger.info(f"Copied {modality} expert weights to MoME")
    
    return model


def freeze_experts(model: MoMESegmenter):
    """Freeze all expert parameters, leave fusion trainable."""
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'experts' in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()
    
    logger.info(f"Frozen expert params: {frozen_params:,}")
    logger.info(f"Trainable fusion params: {trainable_params:,}")


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Prepare inputs for MoME
        inputs = {
            "T1": batch["T1"].to(device),
            "T1ce": batch["T1ce"].to(device),
            "T2": batch["T2"].to(device),
            "FLAIR": batch["FLAIR"].to(device)
        }
        masks = batch["mask"].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs['segmentation'], masks)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
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
            inputs = {
                "T1": batch["T1"].to(device),
                "T1ce": batch["T1ce"].to(device),
                "T2": batch["T2"].to(device),
                "FLAIR": batch["FLAIR"].to(device)
            }
            masks = batch["mask"].to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs['segmentation'], masks)
            
            total_loss += loss.item()
            batch_metrics = metrics.compute_metrics(outputs, {"mask": masks})
            all_metrics.append(batch_metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key]) and m[key] != float('inf')]
        avg_metrics[key] = np.mean(values) if values else 0.0
    avg_metrics['loss'] = total_loss / len(dataloader)
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train fusion network with frozen experts")
    parser.add_argument("--expert_t1", type=str, required=True, help="Path to T1 expert checkpoint")
    parser.add_argument("--expert_t1ce", type=str, required=True, help="Path to T1ce expert checkpoint")
    parser.add_argument("--expert_t2", type=str, required=True, help="Path to T2 expert checkpoint")
    parser.add_argument("--expert_flair", type=str, required=True, help="Path to FLAIR expert checkpoint")
    parser.add_argument("--train_data", type=str, default="data/preprocessed/brats2024_gli_train.h5")
    parser.add_argument("--val_data", type=str, default="data/preprocessed/brats2024_gli_val.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training fusion network on {device}")
    
    # Load pre-trained experts
    logger.info("Loading pre-trained experts...")
    experts = {
        "T1": load_expert(args.expert_t1, device),
        "T1ce": load_expert(args.expert_t1ce, device),
        "T2": load_expert(args.expert_t2, device),
        "FLAIR": load_expert(args.expert_flair, device)
    }
    
    # Create MoME model with expert weights
    model = create_mome_from_experts(experts, device)
    
    # Freeze experts
    freeze_experts(model)
    
    # Data augmentation
    train_transforms = Compose([
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "mask"], prob=0.3, max_k=3),
    ])
    
    # Create datasets
    train_dataset = MultiModalityDataset(args.train_data, transform=train_transforms)
    val_dataset = MultiModalityDataset(args.val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Only train fusion parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    loss_fn = DiceLoss()
    metrics = SegmentationMetrics(num_classes=3, class_names=["WT", "TC", "ET"])
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Checkpoint directory
    checkpoint_dir = Path("experiments/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_dice = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate(model, val_loader, loss_fn, metrics, device)
            
            mean_dice = (val_metrics.get('dice_WT', 0) + 
                        val_metrics.get('dice_TC', 0) + 
                        val_metrics.get('dice_ET', 0)) / 3
            
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"WT: {val_metrics.get('dice_WT', 0):.4f}, "
                       f"TC: {val_metrics.get('dice_TC', 0):.4f}, "
                       f"ET: {val_metrics.get('dice_ET', 0):.4f}")
            logger.info(f"Mean Dice: {mean_dice:.4f}")
            
            scheduler.step(mean_dice)
            
            if mean_dice > best_dice:
                best_dice = mean_dice
                epochs_without_improvement = 0
                
                # Save full model
                checkpoint_path = checkpoint_dir / "mome_fusion_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'val_metrics': val_metrics
                }, checkpoint_path)
                logger.info(f"✅ Saved best model (Dice: {best_dice:.4f})")
            else:
                epochs_without_improvement += 5
            
            if epochs_without_improvement >= args.patience:
                logger.info(f"Early stopping after {epochs_without_improvement} epochs without improvement")
                break
    
    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Best Mean Dice: {best_dice:.4f}")
    logger.info(f"Checkpoint: {checkpoint_dir / 'mome_fusion_best.pth'}")


if __name__ == "__main__":
    main()
