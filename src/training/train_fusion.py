"""
Train the fusion network with pre-trained frozen experts.

This script:
1. Loads 4 pre-trained modality experts
2. Freezes all expert weights
3. Trains only the fusion / gating network to combine expert outputs

Usage (per-modality H5 files — the default):
    python -m src.training.train_fusion \
        --expert_t1   experiments/checkpoints/experts/expert_T1_best.pth \
        --expert_t1ce experiments/checkpoints/experts/expert_T1ce_best.pth \
        --expert_t2   experiments/checkpoints/experts/expert_T2_best.pth \
        --expert_flair experiments/checkpoints/experts/expert_FLAIR_best.pth

Usage (single combined H5 file):
    python -m src.training.train_fusion \
        --expert_t1   ... \
        --train_data  data/preprocessed/brats2024_gli_train.h5 \
        --val_data    data/preprocessed/brats2024_gli_val.h5
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
from typing import Dict, List, Optional

from src.models.mome_expert import ModalityExpert
from src.models.mome_segmenter import MoMESegmenter
from src.training.loss_functions import DiceLoss
from src.training.metrics import SegmentationMetrics
from src.utils.logger import get_logger

# Enable cuDNN auto-tuner – finds best convolution algorithms for fixed input sizes
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset – supports BOTH per-modality H5 files AND single combined H5 file
# ---------------------------------------------------------------------------

class MultiModalityDataset(Dataset):
    """Dataset that returns all 4 modalities for fusion training.

    Supports two data layouts:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Mode A – Per-modality files (DEFAULT)                          │
    │    brats2024_gli_T1_train.h5    → image shape (1, D, H, W)     │
    │    brats2024_gli_T1ce_train.h5  → image shape (1, D, H, W)     │
    │    brats2024_gli_T2_train.h5    → image shape (1, D, H, W)     │
    │    brats2024_gli_FLAIR_train.h5 → image shape (1, D, H, W)     │
    │    (mask is read from T1 file)                                  │
    │                                                                 │
    │  Mode B – Combined single file                                  │
    │    brats2024_gli_train.h5       → image shape (4, D, H, W)     │
    └─────────────────────────────────────────────────────────────────┘

    Optimised for speed:
    - Lazy HDF5 handles that stay open across calls (avoids per-sample open/close)
    - Fast numpy-based augmentation (no MONAI overhead)
    - Contiguous memory layout for fast GPU transfer
    """

    MODALITY_INDEX = {"T1": 0, "T1ce": 1, "T2": 2, "FLAIR": 3}
    MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]

    def __init__(
        self,
        h5_path: Optional[str] = None,
        augment: bool = False,
        max_crops: Optional[int] = None,
        # Per-modality paths (Mode A)
        t1_path: Optional[str] = None,
        t1ce_path: Optional[str] = None,
        t2_path: Optional[str] = None,
        flair_path: Optional[str] = None,
    ):
        self.augment = augment

        # ── Determine mode ───────────────────────────────────────────────────
        per_modality_paths = {
            "T1": t1_path, "T1ce": t1ce_path, "T2": t2_path, "FLAIR": flair_path
        }
        using_per_modality = all(p is not None for p in per_modality_paths.values())

        if using_per_modality:
            self.mode = "per_modality"
            self.modality_paths = per_modality_paths
            # Count crops from T1 file (all files should have equal crop count)
            with h5py.File(t1_path, "r") as f:
                self.num_crops = f.attrs.get(
                    "num_crops",
                    len([k for k in f.keys() if k.startswith("crop")])
                )
            logger.info(
                f"Dataset (per-modality mode): {self.num_crops} crops\n"
                f"  T1:    {t1_path}\n"
                f"  T1ce:  {t1ce_path}\n"
                f"  T2:    {t2_path}\n"
                f"  FLAIR: {flair_path}"
            )
            # Lazy handles – one per modality
            self.h5_files: Dict[str, Optional[h5py.File]] = {
                m: None for m in self.MODALITIES
            }
        elif h5_path is not None:
            self.mode = "combined"
            self.h5_path = h5_path
            with h5py.File(h5_path, "r") as f:
                self.num_crops = f.attrs.get(
                    "num_crops",
                    len([k for k in f.keys() if k.startswith("crop")])
                )
            logger.info(f"Dataset (combined mode): {self.num_crops} crops from {h5_path}")
            self.h5_file: Optional[h5py.File] = None
        else:
            raise ValueError(
                "Provide either --train_data (combined H5) OR all four "
                "--t1_data / --t1ce_data / --t2_data / --flair_data paths."
            )

        if max_crops and max_crops < self.num_crops:
            self.num_crops = max_crops

    # ── Lazy H5 handles ──────────────────────────────────────────────────────

    def _open_combined(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

    def _open_per_modality(self):
        for mod, path in self.modality_paths.items():
            if self.h5_files[mod] is None:
                self.h5_files[mod] = h5py.File(path, "r")

    # ── Dataset protocol ────────────────────────────────────────────────────

    def __len__(self):
        return self.num_crops

    def __getitem__(self, idx):
        key = f"crop_{idx:06d}"

        if self.mode == "per_modality":
            self._open_per_modality()
            # Each file has shape (1, D, H, W)
            t1    = self.h5_files["T1"][key]["image"][:]
            t1ce  = self.h5_files["T1ce"][key]["image"][:]
            t2    = self.h5_files["T2"][key]["image"][:]
            flair = self.h5_files["FLAIR"][key]["image"][:]
            mask  = self.h5_files["T1"][key]["mask"][:]   # same mask in all files
            # Stack to (4, D, H, W) for augmentation, then split
            image = np.concatenate([t1, t1ce, t2, flair], axis=0)
        else:
            self._open_combined()
            image = self.h5_file[key]["image"][:]   # (4, D, H, W)
            mask  = self.h5_file[key]["mask"][:]    # (3, D, H, W)

        # ── Numpy augmentation (fast; no MONAI overhead) ─────────────────────
        if self.augment:
            for axis in (1, 2, 3):      # D, H, W
                if np.random.rand() > 0.5:
                    image = np.flip(image, axis=axis).copy()
                    mask  = np.flip(mask,  axis=axis).copy()
            k = np.random.randint(0, 4)
            if k:
                image = np.rot90(image, k=k, axes=(2, 3)).copy()
                mask  = np.rot90(mask,  k=k, axes=(2, 3)).copy()

        # ── Return per-modality tensors ──────────────────────────────────────
        def _to_tensor(arr):
            return torch.from_numpy(np.ascontiguousarray(arr)).float()

        return {
            "T1":    _to_tensor(image[0:1]),
            "T1ce":  _to_tensor(image[1:2]),
            "T2":    _to_tensor(image[2:3]),
            "FLAIR": _to_tensor(image[3:4]),
            "mask":  _to_tensor(mask),
        }

    def __del__(self):
        """Close any open HDF5 handles."""
        if self.mode == "combined":
            try:
                if self.h5_file is not None:
                    self.h5_file.close()
            except Exception:
                pass
        else:
            for f in self.h5_files.values():
                try:
                    if f is not None:
                        f.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Expert / model helpers
# ---------------------------------------------------------------------------

def load_expert(checkpoint_path: str, modality: str,
                device: torch.device) -> ModalityExpert:
    """Load a pre-trained ModalityExpert from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    ckpt_modality = checkpoint.get("modality", modality)

    expert = ModalityExpert(
        modality=ckpt_modality,
        in_channels=1,
        num_classes=3,
        base_channels=32,
        depth=4,
    )
    expert.load_state_dict(checkpoint["model_state_dict"])
    expert.to(device)

    best_dice = checkpoint.get("best_dice", 0.0)
    logger.info(f"Loaded {ckpt_modality} expert  (best Dice: {best_dice:.4f})")
    return expert


def create_mome_from_experts(experts: Dict[str, ModalityExpert],
                              device: torch.device) -> MoMESegmenter:
    """Instantiate MoMESegmenter and copy pre-trained expert weights into it."""
    model = MoMESegmenter(
        modalities=["T1", "T1ce", "T2", "FLAIR"],
        in_channels=1,
        num_classes=3,
        base_channels=32,
        depth=4,
    ).to(device)

    for modality, expert in experts.items():
        model.experts[modality].load_state_dict(expert.state_dict())
        logger.info(f"Copied {modality} expert weights → MoME model")

    return model


def freeze_experts(model: MoMESegmenter):
    """Freeze all expert sub-modules; leave gating / fusion trainable."""
    frozen = trainable = 0
    for name, param in model.named_parameters():
        if "experts" in name:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()

    logger.info(f"Frozen expert params:    {frozen:,}")
    logger.info(f"Trainable fusion params: {trainable:,}")


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scaler, loss_fn, device, epoch):
    """One training epoch with AMP + GradScaler."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]")
    for batch in pbar:
        inputs = {
            "T1":    batch["T1"].to(device, non_blocking=True),
            "T1ce":  batch["T1ce"].to(device, non_blocking=True),
            "T2":    batch["T2"].to(device, non_blocking=True),
            "FLAIR": batch["FLAIR"].to(device, non_blocking=True),
        }
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = loss_fn(outputs["segmentation"], masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def validate(model, dataloader, loss_fn, metrics, device):
    """Validation loop – returns a dict of averaged metrics."""
    model.eval()
    total_loss = 0.0
    all_metrics: List[Dict] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs = {
                "T1":    batch["T1"].to(device, non_blocking=True),
                "T1ce":  batch["T1ce"].to(device, non_blocking=True),
                "T2":    batch["T2"].to(device, non_blocking=True),
                "FLAIR": batch["FLAIR"].to(device, non_blocking=True),
            }
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = loss_fn(outputs["segmentation"], masks)

            total_loss += loss.item()
            batch_metrics = metrics.compute_metrics(outputs, {"mask": masks})
            all_metrics.append(batch_metrics)

    # Average across batches, ignoring NaN / Inf
    avg: Dict = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics
                  if not np.isnan(m[key]) and m[key] != float("inf")]
        avg[key] = float(np.mean(values)) if values else 0.0
    avg["loss"] = total_loss / len(dataloader)
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the MoME+ fusion / gating network with frozen experts"
    )

    # ── Expert checkpoints ──────────────────────────────────────────────────
    parser.add_argument("--expert_t1",    type=str,
        default="experiments/checkpoints/experts/expert_T1_best.pth")
    parser.add_argument("--expert_t1ce", type=str,
        default="experiments/checkpoints/experts/expert_T1ce_best.pth")
    parser.add_argument("--expert_t2",   type=str,
        default="experiments/checkpoints/experts/expert_T2_best.pth")
    parser.add_argument("--expert_flair", type=str,
        default="experiments/checkpoints/experts/expert_FLAIR_best.pth")

    # ── Data – combined file (Mode B) ───────────────────────────────────────
    parser.add_argument("--train_data", type=str, default=None,
        help="Combined 4-channel training H5 file. If omitted, use per-modality files.")
    parser.add_argument("--val_data",   type=str, default=None,
        help="Combined 4-channel validation H5 file. If omitted, use per-modality files.")

    # ── Data – per-modality files (Mode A, DEFAULT) ─────────────────────────
    parser.add_argument("--t1_train",    type=str,
        default="../data/preprocessed/brats2024_gli_T1_train.h5")
    parser.add_argument("--t1ce_train",  type=str,
        default="../data/preprocessed/brats2024_gli_T1ce_train.h5")
    parser.add_argument("--t2_train",    type=str,
        default="../data/preprocessed/brats2024_gli_T2_train.h5")
    parser.add_argument("--flair_train", type=str,
        default="../data/preprocessed/brats2024_gli_FLAIR_train.h5")

    parser.add_argument("--t1_val",    type=str,
        default="../data/preprocessed/brats2024_gli_T1_val.h5")
    parser.add_argument("--t1ce_val",  type=str,
        default="../data/preprocessed/brats2024_gli_T1ce_val.h5")
    parser.add_argument("--t2_val",    type=str,
        default="../data/preprocessed/brats2024_gli_T2_val.h5")
    parser.add_argument("--flair_val", type=str,
        default="../data/preprocessed/brats2024_gli_FLAIR_val.h5")

    # ── Hyper-parameters ────────────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=4,
        help="Reduce to 2 if CUDA out-of-memory")
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=15,
        help="Early-stopping: number of validation events without improvement")
    parser.add_argument("--val_freq",     type=int,   default=5,
        help="Run validation every N epochs")

    # ── Runtime ─────────────────────────────────────────────────────────────
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0,
        help="DataLoader workers. Keep at 0 on Windows + HDF5.")
    parser.add_argument("--output_dir",  type=str,
        default="experiments/checkpoints",
        help="Directory for fusion checkpoints")

    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU : {torch.cuda.get_device_name(device)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # ── Validate expert checkpoint paths ────────────────────────────────────
    expert_paths = {
        "T1":    args.expert_t1,
        "T1ce":  args.expert_t1ce,
        "T2":    args.expert_t2,
        "FLAIR": args.expert_flair,
    }
    for mod, path in expert_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Expert checkpoint for {mod} not found: {path}\n"
                "Place the checkpoint at that path and re-run."
            )

    # ── Load experts ─────────────────────────────────────────────────────────
    logger.info("Loading pre-trained experts …")
    experts = {mod: load_expert(path, mod, device)
               for mod, path in expert_paths.items()}

    # ── Build MoME model, copy weights, freeze experts ───────────────────────
    model = create_mome_from_experts(experts, device)
    del experts     # free redundant copies

    freeze_experts(model)

    # ── Build datasets ───────────────────────────────────────────────────────
    # Prefer combined file if explicitly provided, otherwise use per-modality
    if args.train_data is not None:
        train_dataset = MultiModalityDataset(h5_path=args.train_data, augment=True)
        val_dataset   = MultiModalityDataset(h5_path=args.val_data,   augment=False)
        logger.info("Data mode: combined H5 file")
    else:
        train_dataset = MultiModalityDataset(
            augment=True,
            t1_path=args.t1_train,   t1ce_path=args.t1ce_train,
            t2_path=args.t2_train,   flair_path=args.flair_train,
        )
        val_dataset = MultiModalityDataset(
            augment=False,
            t1_path=args.t1_val,   t1ce_path=args.t1ce_val,
            t2_path=args.t2_val,   flair_path=args.flair_val,
        )
        logger.info("Data mode: per-modality H5 files")

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ num_workers=0  → required on Windows; HDF5 is NOT fork-safe.       │
    # │ pin_memory=False → pinning an HDF5-backed dataset slows things.    │
    # │ drop_last=True   → keeps batch stats stable for BatchNorm layers.  │
    # └─────────────────────────────────────────────────────────────────────┘
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=False,
    )
    logger.info(f"Train batches: {len(train_loader)},  Val batches: {len(val_loader)}")

    # ── Optimizer (fusion+gating params only) ───────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Halve LR when validation Mean-Dice stagnates for 5 consecutive val events
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )

    # ── Loss, metrics, AMP scaler ────────────────────────────────────────────
    loss_fn = DiceLoss()
    metrics = SegmentationMetrics(num_classes=3, class_names=["WT", "TC", "ET"])
    scaler  = torch.amp.GradScaler("cuda")

    # ── Checkpoint dir ───────────────────────────────────────────────────────
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    best_dice = 0.0
    val_events_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device, epoch
        )
        logger.info(f"Train Loss: {train_loss:.4f}")

        # ── Periodic validation ──────────────────────────────────────────────
        if epoch % args.val_freq == 0 or epoch == 1:
            val_metrics = validate(model, val_loader, loss_fn, metrics, device)

            mean_dice = (
                val_metrics.get("dice_WT", 0.0)
                + val_metrics.get("dice_TC", 0.0)
                + val_metrics.get("dice_ET", 0.0)
            ) / 3.0

            logger.info(
                f"Val Loss: {val_metrics['loss']:.4f}  |  "
                f"WT={val_metrics.get('dice_WT', 0):.4f}  "
                f"TC={val_metrics.get('dice_TC', 0):.4f}  "
                f"ET={val_metrics.get('dice_ET', 0):.4f}  "
                f"→ Mean={mean_dice:.4f}"
            )
            logger.info(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            scheduler.step(mean_dice)

            if mean_dice > best_dice:
                best_dice = mean_dice
                val_events_no_improve = 0
                checkpoint_path = checkpoint_dir / "mome_fusion_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_dice": best_dice,
                        "val_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
                logger.info(f"✅ Best model saved  (Mean Dice: {best_dice:.4f})")
            else:
                val_events_no_improve += 1
                logger.info(
                    f"No improvement – {val_events_no_improve} val event(s) "
                    f"(patience={args.patience})"
                )

            if val_events_no_improve >= args.patience:
                logger.info(
                    f"Early stopping after {val_events_no_improve} consecutive "
                    "val events without improvement."
                )
                break

        # ── Save 'last' checkpoint every epoch (for resuming) ────────────────
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_dice": best_dice,
            },
            checkpoint_dir / "mome_fusion_last.pth",
        )

    logger.info("\n=== Training Complete ===")
    logger.info(f"Best Mean Dice : {best_dice:.4f}")
    logger.info(f"Best checkpoint: {checkpoint_dir / 'mome_fusion_best.pth'}")


if __name__ == "__main__":
    main()
