# Implementation Plan: MoME Architecture Separate Modality Training

## Goal

Restructure the existing MoME+ architecture to match the paper's approach where each modality expert (T1, T1ce, T2, FLAIR) is trained separately, followed by a fusion/gating network training phase. The implementation will be optimized for RTX 3080 10GB GPU training.

## User Review Required

> [!IMPORTANT]
> **Batch Size Efficiency - Primary Motivation**: Training each expert separately allows **much larger batch sizes**:
> - **Joint training** (4 modalities): batch_size = 1-2 (very inefficient)
> - **Separate expert training** (1 modality): batch_size = 4-8+ (4x improvement!)
> 
> **Configuration for RTX 3080 10GB**:
> - Batch size: **6-8** per expert (vs 1-2 for joint training)
> - Crop size: [128, 128, 128] (adjust if needed)
> - Base channels: 32 (keep original, memory allows with single modality)
> - Depth: 4 (keep original)
> - Mixed precision training (FP16) for extra memory
> - Gradient accumulation if batch size still limited

> [!WARNING]
> **Training Philosophy Change**: Instead of training all experts jointly with the gating network (current approach), we'll train:
> 1. Each modality expert independently to learn modality-specific features
> 2. Then train the gating/fusion network with frozen expert encoders
>
> This matches the paper's approach and reduces memory requirements during training.

## Proposed Changes

### 1. Configuration Files

#### [NEW] [expert_train_config.yaml](file:///c:/Users/Farhan/Desktop/FYP/An-Expert-Guided-Multimodal-AI-Ecosystem/configs/expert_train_config.yaml)

Base configuration for individual expert training optimized for larger batches:
- **Batch size: 6-8** (key advantage of separate training!)
- Cropped patches: [128, 128, 128] (standard for BraTS)
- Base channels: 32 (original capacity maintained)
- Depth: 4 (original depth maintained)
- Mixed precision training (FP16) enabled
- Gradient accumulation: 2 (effective batch = 12-16)
- Epochs: 100-150 per expert

#### [NEW] [fusion_train_config.yaml](file:///c:/Users/Farhan/Desktop/FYP/An-Expert-Guided-Multimodal-AI-Ecosystem/configs/fusion_train_config.yaml)

Configuration for training the gating/fusion network:
- Load pre-trained expert weights
- Freeze expert encoders
- Train only gating network and fusion layers
- Smaller learning rate (1e-5)
- Epochs: 50

---

### 2. Training Scripts

#### [NEW] [train_t1_expert.py](file:///c:/Users/Farhan/Desktop/FYP/An-Expert-Guided-Multimodal-AI-Ecosystem/train_t1_expert.py)

Individual training script for T1 modality expert:
- Initialize single `ModalityExpert` for T1
- Load only T1 modality data
- Train with Dice + CE loss
- Save checkpoints to `experiments/checkpoints/t1_expert/`
- Track metrics specific to T1

#### [NEW] [train_t1ce_expert.py](file:///c:/Users/Farhan/Desktop/FYP/An-Expert-Guided-Multimodal-AI-Ecosystem/train_t1ce_expert.py)

Individual training script for T1ce modality expert (similar structure to T1)

#### [NEW] [train_t2_expert.py](file:///c:/Users/Farhan/Desktop/FYP/An-Expert-Guided-Multimodal-AI-Ecosystem/train_t2_expert.py)

Individual training script for T2 modality expert (similar structure to T1)

#### [NEW] [train_flair_expert.py](file:///c:/Users/Farhan/Desktop/FYP/An-Expert-Guided-Multimodal-AI-Ecosystem/train_flair_expert.py)

Individual training script for FLAIR modality expert (similar structure to T1)

#### [NEW] [train_fusion_network.py](file:///c:/Users/Farhan/Desktop/FYP/An-Expert-Guided-Multimodal-AI-Ecosystem/train_fusion_network.py)

Training script for the gating/fusion network:
- Load all 4 pre-trained expert models
- Create `MoMESegmenter` with pre-trained experts
- Freeze expert encoder weights
- Train only:
  - `HierarchicalGatingNetwork`
  - `ExpertFusion` module
  - Expert decoder layers (optional, can be frozen)
- Multi-modal input required
- Save complete model with optimized gating

---

### 3. Model Architecture Adjustments

#### [MODIFY] [src/models/mome_expert.py](file:///c:/Users/Farhan/Desktop/FYP/An-Expert-Guided-Multimodal-AI-Ecosystem/src/models/mome_expert.py)

Add method to `ModalityExpert`:
- `load_pretrained_encoder()` - Load only encoder weights
- Add gradient checkpointing support
- Add method to return parameter count for memory estimation

#### [MODIFY] [src/models/mome_segmenter.py](file:///c:/Users/Farhan/Desktop/FYP/An-Expert-Guided-Multimodal-AI-Ecosystem/src/models/mome_segmenter.py)

Add methods to `MoMESegmenter`:
- `load_pretrained_experts()` - Load individual expert checkpoints
- `freeze_all_experts()` - Freeze all expert parameters
- `unfreeze_gating_only()` - Unfreeze only gating/fusion parameters
- Support mixed precision training

---

### 4. Data Loading Utilities

#### [NEW] [src/data/single_modality_dataset.py](file:///c:/Users/Farhan/Desktop/FYP/An-Expert-Guided-Multimodal-AI-Ecosystem/src/data/single_modality_dataset.py)

Dataset class for loading single modality data:
- Extends existing BraTS dataset
- Returns only specified modality + ground truth
- Memory-efficient loading
- Same preprocessing as multi-modal

---

### 5. Training Workflow Helper

#### [NEW] [scripts/train_all_experts.py](file:///c:/Users/Farhan/Desktop/FYP/An-Expert-Guided-Multimodal-AI-Ecosystem/scripts/train_all_experts.py)

Convenience script to train all experts sequentially:
- Train T1 → T1ce → T2 → FLAIR
- Monitor GPU memory usage
- Aggregate metrics
- Optional: parallel training if multiple GPUs available

## Verification Plan

### Automated Tests

1. **Model Memory Test**
   ```bash
   python -m pytest tests/test_model_memory.py::test_expert_memory_footprint
   ```
   - Verify single expert fits in 10GB VRAM
   - Verify fusion training fits in 10GB VRAM

2. **Expert Training Test**
   ```bash
   python train_t1_expert.py --config configs/expert_train_config.yaml --epochs 2 --validate_only
   ```
   - Quick 2-epoch test run for T1 expert
   - Verify training loop works
   - Check checkpoint saving

3. **Fusion Training Test**
   ```bash
   python train_fusion_network.py --config configs/fusion_train_config.yaml --epochs 2 --validate_only
   ```
   - Quick 2-epoch test run for fusion network
   - Verify expert loading and freezing
   - Check gradient flow only to gating network

### Manual Verification

1. **Check GPU Memory Usage**
   - Run `nvidia-smi` during training
   - Ensure memory usage < 9.5GB (leaving headroom)
   - User should monitor and report if OOM occurs

2. **Verify Expert Checkpoints**
   - Check `experiments/checkpoints/t1_expert/best_model.pth` exists
   - Verify checkpoint contains expert state_dict
   - Load and test inference on single sample

3. **Verify Complete Pipeline**
   - Train one expert (T1) to completion
   - Train fusion network with T1 expert loaded
   - Run inference and verify output shapes match expected [B, 3, D, H, W]

## Training Order

1. Train T1 expert (~6-8 hours on RTX 3080)
2. Train T1ce expert (~6-8 hours)
3. Train T2 expert (~6-8 hours)
4. Train FLAIR expert (~6-8 hours)
5. Train fusion network with all 4 experts (~4-6 hours)

**Total training time: ~28-38 hours**

## Expected Outcomes

- 4 separate expert model checkpoints
- 1 complete MoME model with optimized gating
- Training logs for each component
- Reduced memory footprint per training run
- More modular architecture for future experiments
