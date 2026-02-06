# MoME+ Training Quick Start

This guide covers everything needed to train the MoME+ brain tumor segmentation model.

---

## 1. Dataset Setup

### BraTS 2024 GLI Dataset Structure

Download from [Synapse](https://www.synapse.org/#!Synapse:syn51514132) and place in:

```
<your_data_path>/Brats2024/BratsGLI/
├── training_data1_v2/
│   ├── BraTS-GLI-00000-000/
│   │   ├── BraTS-GLI-00000-000-t1n.nii.gz
│   │   ├── BraTS-GLI-00000-000-t1c.nii.gz
│   │   ├── BraTS-GLI-00000-000-t2w.nii.gz
│   │   ├── BraTS-GLI-00000-000-t2f.nii.gz
│   │   └── BraTS-GLI-00000-000-seg.nii.gz
│   └── ... (more cases)
└── training_data_additional/
    └── ... (more cases)
```

### Update Paths in Preprocessing Script

Edit `scripts/preprocess_brats_crops.py` lines 236-239:

```python
train_dirs = [
    "./../../dataset_script/brats_data/brats2024/brats2024-brats-gli-trainingdata",
    "./../../dataset_script/brats_data/brats2024/brats2024-brats-gli-additionaltrainingdata"
]
```

---

## 2. Environment Setup

```powershell
cd <project_directory>
.\.venv\Scripts\activate
```

---

## 3. Preprocessing (~90 minutes)

Creates 8 separate HDF5 files (one per modality per split) for optimal training speed.

```powershell
.\.venv\Scripts\python.exe scripts/preprocess_brats_crops.py --crop_size 96 --crops_per_case 10
```

**Output files** in `./../data/preprocessed/`:
- `brats2024_gli_T1_train.h5`, `brats2024_gli_T1_val.h5`
- `brats2024_gli_T1ce_train.h5`, `brats2024_gli_T1ce_val.h5`
- `brats2024_gli_T2_train.h5`, `brats2024_gli_T2_val.h5`
- `brats2024_gli_FLAIR_train.h5`, `brats2024_gli_FLAIR_val.h5`

---

## 4. Train Individual Experts

Train each modality expert separately. **Use batch_size 5** (optimal for most GPUs).

```powershell
# Generic command template
# .\.venv\Scripts\python.exe diagnostic_files/train_fixed.py --modality [MODALITY] --epochs 100 --batch_size 5

# T1 Expert (~2-3 hours)
.\.venv\Scripts\python.exe diagnostic_files/train_fixed.py --modality T1 --epochs 100 --batch_size 5

# T1ce Expert (~2-3 hours)
.\.venv\Scripts\python.exe diagnostic_files/train_fixed.py --modality T1ce --epochs 100 --batch_size 5

# T2 Expert (~2-3 hours)
.\.venv\Scripts\python.exe diagnostic_files/train_fixed.py --modality T2 --epochs 100 --batch_size 5

# FLAIR Expert (~2-3 hours)
.\.venv\Scripts\python.exe diagnostic_files/train_fixed.py --modality FLAIR --epochs 100 --batch_size 5
```

**Checkpoints saved to:** `experiments/checkpoints/experts/expert_{modality}_best.pth`

### Resuming Training

If training is interrupted, you can resume from the last checkpoint (best or latest):

```powershell
# Continue training T1ce from best checkpoint
.\.venv\Scripts\python.exe diagnostic_files/train_fixed.py --modality T1ce --epochs 100 --batch_size 5 --resume "experiments/checkpoints/experts/expert_T1ce_best.pth"

# Continue from the absolute last saved epoch (useful if not the best)
.\.venv\Scripts\python.exe diagnostic_files/train_fixed.py --modality T1ce --epochs 100 --batch_size 5 --resume "experiments/checkpoints/experts/expert_T1ce_last.pth"
```

---

## 5. Train Fusion Network

After all 4 experts are trained:

```powershell
.\.venv\Scripts\python.exe -m src.training.train_fusion `
    --expert_t1 experiments/checkpoints/experts/expert_T1_best.pth `
    --expert_t1ce experiments/checkpoints/experts/expert_T1ce_best.pth `
    --expert_t2 experiments/checkpoints/experts/expert_T2_best.pth `
    --expert_flair experiments/checkpoints/experts/expert_FLAIR_best.pth `
    --epochs 50 --batch_size 4
```

---

## 6. Continual Learning (Optional)

For BratsMEN dataset after base training:

```powershell
# First preprocess MEN dataset
.\.venv\Scripts\python.exe scripts/preprocess_bratsmen.py --crop_size 96 --crops_per_case 10

# Then run continual learning
.\.venv\Scripts\python.exe -m src.training.continual_trainer `
    --config configs/train_config_continual_men.yaml `
    --base_model experiments/checkpoints/mome_fusion_best.pth `
    --replay --epochs 50 --batch_size 4 --ewc_lambda 5000
```

---

## Checkpoint Summary

| File | Description |
|------|-------------|
| `experts/expert_T1_best.pth` | Trained T1 expert |
| `experts/expert_T1ce_best.pth` | Trained T1ce expert |
| `experts/expert_T2_best.pth` | Trained T2 expert |
| `experts/expert_FLAIR_best.pth` | Trained FLAIR expert |
| `mome_fusion_best.pth` | Full MoME with trained fusion |
