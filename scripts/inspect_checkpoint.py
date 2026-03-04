"""Quick script to inspect what's in the fusion checkpoint."""
import torch

ckpt = torch.load('experiments/checkpoints/experts/mome_fusion_best.pth', 
                   map_location='cpu', weights_only=False)

print("Top-level keys:", list(ckpt.keys()))
sd = ckpt.get('model_state_dict', ckpt)
print(f"\nTotal state_dict keys: {len(sd)}")

expert_keys = [k for k in sd if k.startswith('experts.')]
gating_keys = [k for k in sd if 'gating' in k]
fusion_keys = [k for k in sd if 'expert_fusion' in k]

print(f"Expert keys: {len(expert_keys)}")
print(f"Gating keys: {len(gating_keys)}")
print(f"Fusion keys: {len(fusion_keys)}")

print("\nFirst 15 keys:")
for k in list(sd.keys())[:15]:
    print(f"  {k}: {sd[k].shape}")

# Also check an individual expert checkpoint
import os
exp_path = 'experiments/checkpoints/experts/expert_T1_best.pth'
if os.path.exists(exp_path):
    e_ckpt = torch.load(exp_path, map_location='cpu', weights_only=False)
    e_sd = e_ckpt.get('model_state_dict', e_ckpt)
    print(f"\nExpert T1 checkpoint keys: {len(e_sd)}")
    print("First 5:")
    for k in list(e_sd.keys())[:5]:
        print(f"  {k}: {e_sd[k].shape}")
