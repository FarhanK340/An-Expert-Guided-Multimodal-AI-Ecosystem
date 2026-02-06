"""
Continual Learning Trainer for MoME+ architecture.

Uses existing EWC and ReplayBuffer modules from continual_learning.py.
Trains on new task (BratsMEN) while preserving knowledge from old task (BratsGLI).
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, List

# Enable cuDNN auto-tuner for fixed input sizes (can speed up training)
# Set to True if input sizes are fixed for additional speedup
torch.backends.cudnn.benchmark = False
# Use existing continual learning modules
from src.models.continual_learning import EWC, ReplayBuffer, ContinualLearningWrapper
from src.models.mome_segmenter import MoMESegmenter
from src.training.preprocessed_dataset import PreprocessedBraTSDataset
from src.training.metrics import SegmentationMetrics
from src.training.loss_functions import DiceLoss
from src.utils.config_parser import load_config
from src.utils.logger import get_logger

# MONAI transforms for data augmentation
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandGaussianNoised,
    RandGaussianSmoothd, RandAffined
)

logger = get_logger(__name__)


def prepare_input_for_model(images: torch.Tensor, task: str = "gli") -> Dict[str, torch.Tensor]:
    """
    Prepare input dictionary for MoME+ model.
    
    Args:
        images: Input tensor
        task: "gli" (4 modalities) or "men" (T1c only)
    
    Returns:
        Dictionary of modality inputs
    """
    if task == "gli":
        # BratsGLI has 4 modalities
        return {
            'T1': images[:, 0:1],
            'T1ce': images[:, 1:2],
            'T2': images[:, 2:3],
            'FLAIR': images[:, 3:4]
        }
    else:
        # BratsMEN has only T1c (passed as single channel)
        return {
            'T1': torch.zeros_like(images),
            'T1ce': images,  # T1c data
            'T2': torch.zeros_like(images),
            'FLAIR': torch.zeros_like(images)
        }


def expand_output_layer(model: MoMESegmenter, old_num_classes: int, new_num_classes: int, device: torch.device):
    """
    Expand the output layer from old_num_classes to new_num_classes.
    Preserves existing weights and initializes new class weights.
    """
    for modality, expert in model.experts.items():
        old_conv = expert.final_conv  # Changed from output_conv
        
        new_conv = nn.Conv3d(
            old_conv.in_channels,
            new_num_classes,
            kernel_size=old_conv.kernel_size,
            padding=old_conv.padding
        ).to(device)
        
        with torch.no_grad():
            new_conv.weight[:old_num_classes] = old_conv.weight
            new_conv.bias[:old_num_classes] = old_conv.bias
            nn.init.kaiming_normal_(new_conv.weight[old_num_classes:])
            new_conv.bias[old_num_classes:].zero_()
        
        expert.final_conv = new_conv  # Changed from output_conv
        logger.info(f"Expanded {modality} expert: {old_num_classes} -> {new_num_classes} classes")
    
    model.num_classes = new_num_classes


def freeze_experts(model: MoMESegmenter, experts_to_freeze: List[str]):
    """Freeze specified expert networks."""
    for modality in experts_to_freeze:
        if modality in model.experts:
            for param in model.experts[modality].parameters():
                param.requires_grad = False
            logger.info(f"Froze expert: {modality}")


def compute_fisher_for_mome(model: MoMESegmenter, dataloader: DataLoader, 
                            device: torch.device, num_samples: int = 200) -> Dict[str, torch.Tensor]:
    """Compute Fisher Information specifically for MoME model with proper input format."""
    logger.info(f"Computing Fisher Information (custom for MoME)...")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model.train()  # Need gradients
    
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    sample_count = 0
    
    pbar = tqdm(dataloader, desc="Fisher computation", total=num_samples // dataloader.batch_size + 1)
    for batch in pbar:
        if sample_count >= num_samples:
            break
            
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Pad GLI masks from 3 to 4 channels (add zero GTV channel) if model is expanded
        if masks.shape[1] == 3:
            zero_channel = torch.zeros_like(masks[:, :1])
            masks = torch.cat([masks, zero_channel], dim=1)
        
        # Prepare input for MoME (GLI has 4 modalities)
        x = prepare_input_for_model(images, task="gli")
        
        model.zero_grad()
        
        # Use AMP for speed
        with torch.cuda.amp.autocast():
            outputs = model(x)
            # Compute loss
            import torch.nn.functional as F
            loss = F.binary_cross_entropy_with_logits(outputs['segmentation'], masks)
        
        # Backward (no scaler needed for Fisher, just need gradients)
        loss.backward()
        
        # Accumulate squared gradients
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.pow(2)
        
        sample_count += images.size(0)
        pbar.set_postfix({'samples': sample_count})
    
    pbar.close()
    
    # Normalize
    for n in fisher:
        fisher[n] = fisher[n] / max(sample_count, 1)
    
    logger.info(f"Fisher Information computed from {sample_count} samples")
    return fisher


def train_epoch(model: MoMESegmenter,
                new_dataloader: DataLoader,
                old_dataloader: Optional[DataLoader],
                optimizer: torch.optim.Optimizer,
                loss_fn: nn.Module,
                ewc: Optional[EWC],
                replay_buffer: Optional[ReplayBuffer],
                device: torch.device,
                use_amp: bool = True) -> Dict[str, float]:
    """Train for one epoch with EWC and replay."""
    model.train()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    total_loss = 0.0
    total_ewc_loss = 0.0
    num_batches = 0
    
    # Iterator for old data replay
    old_iter = iter(old_dataloader) if old_dataloader else None
    
    pbar = tqdm(new_dataloader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Prepare MEN input (T1c only)
        x = prepare_input_for_model(images, task="men")
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(x)
            loss = loss_fn(outputs['segmentation'], masks)
            
            # Replay old task samples
            if old_iter is not None:
                try:
                    old_batch = next(old_iter)
                except StopIteration:
                    old_iter = iter(old_dataloader)
                    old_batch = next(old_iter)
                
                old_images = old_batch['image'].to(device)
                old_masks = old_batch['mask'].to(device)
                
                # Pad GLI masks from 3 to 4 channels (add zero GTV channel)
                if old_masks.shape[1] == 3:
                    zero_channel = torch.zeros_like(old_masks[:, :1])
                    old_masks = torch.cat([old_masks, zero_channel], dim=1)
                
                old_x = prepare_input_for_model(old_images, task="gli")
                
                old_outputs = model(old_x)
                old_loss = loss_fn(old_outputs['segmentation'], old_masks)
                loss = loss + 0.2 * old_loss  # 20% replay weight
            
            # EWC penalty
            ewc_loss_val = 0.0
            if ewc is not None:
                ewc_loss = ewc.get_ewc_loss()
                loss = loss + ewc_loss
                ewc_loss_val = ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_ewc_loss += ewc_loss_val
        num_batches += 1
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'EWC': f'{ewc_loss_val:.4f}'})
    
    return {'loss': total_loss / num_batches, 'ewc_loss': total_ewc_loss / num_batches}


@torch.no_grad()
def validate(model: MoMESegmenter,
             dataloader: DataLoader,
             loss_fn: nn.Module,
             metrics: SegmentationMetrics,
             device: torch.device,
             task: str = "men") -> Dict[str, float]:
    """Validate on a dataset."""
    model.eval()
    total_loss = 0.0
    all_metrics = []
    
    for batch in tqdm(dataloader, desc=f"Validating ({task})"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Pad GLI masks from 3 to 4 channels (add zero GTV channel)
        if task == "gli" and masks.shape[1] == 3:
            zero_channel = torch.zeros_like(masks[:, :1])
            masks = torch.cat([masks, zero_channel], dim=1)
        
        x = prepare_input_for_model(images, task=task)
        
        with torch.cuda.amp.autocast():
            outputs = model(x)
            loss = loss_fn(outputs['segmentation'], masks)
        
        total_loss += loss.item()
        batch_metrics = metrics.compute_metrics(outputs, {'mask': masks})
        all_metrics.append(batch_metrics)
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key]) and m[key] != float('inf')]
        avg_metrics[key] = np.mean(values) if values else 0.0
    avg_metrics['loss'] = total_loss / len(dataloader)
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Continual Learning Trainer")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base GLI model or resume checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--ewc_lambda", type=float, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--replay", action="store_true", help="Use replay buffer")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (skip Fisher/expand)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Start epoch for training loop
    start_epoch = 0
    
    if args.resume:
        # RESUME MODE: Load already-expanded 4-class model
        logger.info(f"Resuming from checkpoint: {args.base_model}")
        model = MoMESegmenter(
            modalities=["T1", "T1ce", "T2", "FLAIR"],
            in_channels=1,
            num_classes=4,  # Already expanded
            base_channels=config.get("model", {}).get("base_channels", 32),
            depth=config.get("model", {}).get("depth", 4)
        ).to(device)
        
        checkpoint = torch.load(args.base_model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")
        
        # Freeze experts (still needed)
        freeze_experts(model, ["T1", "T2", "FLAIR"])
        
        # EWC still enabled - compute Fisher on current model state
        # (we don't need to expand output since it's already 4-class)
        ewc = None  # Will be set after loading GLI data
    else:
        # FRESH START: Load 3-class model and expand
        logger.info(f"Loading base model from {args.base_model}")
        model = MoMESegmenter(
            modalities=["T1", "T1ce", "T2", "FLAIR"],
            in_channels=1,
            num_classes=3,
            base_channels=config.get("model", {}).get("base_channels", 32),
            depth=config.get("model", {}).get("depth", 4)
        ).to(device)
        
        checkpoint = torch.load(args.base_model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded base model weights")
        
        # Expand output: 3 -> 4 classes (do this BEFORE Fisher computation)
        expand_output_layer(model, old_num_classes=3, new_num_classes=4, device=device)
        
        # Freeze T1, T2, FLAIR experts (only train T1ce for MEN)
        freeze_experts(model, ["T1", "T2", "FLAIR"])
        
        # Load GLI data for EWC Fisher computation and replay
        gli_train = PreprocessedBraTSDataset("data/preprocessed/brats2024_gli_train.h5", max_crops=2000)
        gli_train_loader_ewc = DataLoader(gli_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        
        # Initialize EWC with Fisher Information from GLI task
        fisher = compute_fisher_for_mome(model, gli_train_loader_ewc, device, num_samples=500)
        optimal_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        
        ewc = EWC(model, ewc_lambda=args.ewc_lambda)
        ewc.fisher_matrices[0] = fisher
        ewc.optimal_params[0] = optimal_params
        ewc.update_task(1)
    
    # Load GLI data for replay and validation (needed in both modes)
    gli_train = PreprocessedBraTSDataset("data/preprocessed/brats2024_gli_train.h5", max_crops=2000)
    gli_val = PreprocessedBraTSDataset("data/preprocessed/brats2024_gli_val.h5", max_crops=1000)
    gli_train_loader = DataLoader(gli_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    gli_val_loader = DataLoader(gli_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # Initialize EWC for resume mode (compute Fisher on current model state)
    if args.resume and ewc is None:
        logger.info("Computing Fisher for EWC in resume mode...")
        fisher = compute_fisher_for_mome(model, gli_train_loader, device, num_samples=500)
        optimal_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        ewc = EWC(model, ewc_lambda=args.ewc_lambda)
        ewc.fisher_matrices[0] = fisher
        ewc.optimal_params[0] = optimal_params
        ewc.update_task(1)
    
    # Data augmentation for MEN training (lighter version)
    train_transforms = Compose([
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "mask"], prob=0.3, max_k=3),  # Reduced from 0.5
        # Removed GaussianNoise - was too aggressive
    ])
    logger.info("Data augmentation enabled for MEN training (light)")
    
    # Load MEN data (all crops) with augmentation
    men_train = PreprocessedBraTSDataset("data/preprocessed/bratsmen_train.h5", transform=train_transforms)
    men_val = PreprocessedBraTSDataset("data/preprocessed/bratsmen_val.h5")  # No augmentation for validation
    men_train_loader = DataLoader(men_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    men_val_loader = DataLoader(men_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Setup
    loss_fn = DiceLoss()
    metrics = SegmentationMetrics(num_classes=4, class_names=["WT", "TC", "ET", "GTV"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    # Checkpoint paths
    checkpoint_dir = Path("experiments/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_dice = 0.0
    val_frequency = 5  # Validate every 5 epochs
    total_epochs = start_epoch + args.epochs  # Total epochs to reach
    
    for epoch in range(start_epoch, total_epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{total_epochs} ===")
        
        train_metrics = train_epoch(
            model, men_train_loader,
            gli_train_loader if args.replay else None,
            optimizer, loss_fn, ewc, None, device
        )
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, EWC: {train_metrics['ewc_loss']:.4f}")
        
        # Save recent model after every epoch (with backup rotation)
        recent_path = checkpoint_dir / "recent_continual_model.pth"
        backup_path = checkpoint_dir / "recent_continual_model_backup.pth"
        
        if recent_path.exists():
            # Rotate: recent -> backup
            if backup_path.exists():
                backup_path.unlink()
            recent_path.rename(backup_path)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss']
        }, recent_path)
        logger.info(f"Saved recent checkpoint")
        
        # Validation every val_frequency epochs
        if (epoch + 1) % val_frequency == 0 or epoch == args.epochs - 1:
            logger.info(f"--- Validation ---")
            
            # Validate MEN
            men_metrics = validate(model, men_val_loader, loss_fn, metrics, device, task="men")
            logger.info(f"MEN - GTV Dice: {men_metrics.get('dice_GTV', 0):.4f}")
            
            # Validate GLI (check forgetting)
            gli_metrics = validate(model, gli_val_loader, loss_fn, metrics, device, task="gli")
            logger.info(f"GLI - WT: {gli_metrics.get('dice_WT', 0):.4f}, TC: {gli_metrics.get('dice_TC', 0):.4f}, ET: {gli_metrics.get('dice_ET', 0):.4f}")
            
            # Save best (with backup rotation)
            current_dice = (men_metrics.get('dice_GTV', 0) + gli_metrics.get('dice_mean', 0)) / 2
            if current_dice > best_dice:
                best_dice = current_dice
                
                best_path = checkpoint_dir / "best_continual_model.pth"
                best_backup_path = checkpoint_dir / "best_continual_model_backup.pth"
                
                if best_path.exists():
                    if best_backup_path.exists():
                        best_backup_path.unlink()
                    best_path.rename(best_backup_path)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'men_metrics': men_metrics,
                    'gli_metrics': gli_metrics,
                    'combined_dice': current_dice
                }, best_path)
                logger.info(f"Saved best model (Combined Dice: {best_dice:.4f})")
    
    logger.info(f"\n=== Complete! Best Combined Dice: {best_dice:.4f} ===")


if __name__ == "__main__":
    main()
