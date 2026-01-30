# Task: Restructure MoME Architecture for Separate Modality Training

## Overview
Restructure the model architecture to match the MoME paper with separate training for each modality expert and fusion/gating network. **Primary benefit: Enable batch sizes of 6-8 (vs 1-2 for joint training) by training one modality at a time on RTX 3080 10GB.**

## Task Breakdown

### [/] 1. Architecture Analysis
- [x] Review existing MoME+ implementation
- [x] Understand current training pipeline
- [x] Identify configuration requirements
- [ ] Verify GPU memory constraints

### [ ] 2. Model Configuration Updates
- [ ] Create optimized config for RTX 3080 10GB
- [ ] Adjust hyperparameters for memory efficiency
- [ ] Reduce model depth/channels if needed
- [ ] Configure gradient checkpointing

### [ ] 3. Separate Modality Expert Training Scripts
- [ ] Create `train_t1_expert.py`
- [ ] Create `train_t1ce_expert.py`
- [ ] Create `train_t2_expert.py`
- [ ] Create `train_flair_expert.py`
- [ ] Implement individual expert training logic
- [ ] Add checkpoint saving per expert

### [ ] 4. Fusion/Gating Network Training
- [ ] Create `train_fusion_network.py`
- [ ] Load pre-trained expert weights
- [ ] Freeze expert encoders during fusion training
- [ ] Train only gating network and fusion layers

### [ ] 5. Configuration Files
- [ ] Create `configs/expert_train_config.yaml` (base config for experts)
- [ ] Create `configs/fusion_train_config.yaml` (config for fusion)
- [ ] Add GPU memory optimization settings

### [ ] 6. Testing & Verification
- [ ] Test individual expert training
- [ ] Test fusion network training
- [ ] Verify GPU memory usage
- [ ] Validate end-to-end pipeline
