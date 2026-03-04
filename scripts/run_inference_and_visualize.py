"""
End-to-end: run MoME+ inference on a BraTS case and visualize 2D slices.

Uses InferenceEngine from src/inference/inference_engine.py with proper
model loading (expert + fusion checkpoints).

Usage:
    python scripts/run_inference_and_visualize.py --case_dir <path_to_brats_case>
"""

import sys
import numpy as np
import torch
import nibabel as nib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.inference_engine import InferenceEngine
from src.models.mome_segmenter import MoMESegmenter
from src.inference.slice_visualizer import SliceVisualizer

# ---------------------------------------------------------------------------
MODALITY_ORDER = ['T1', 'T1ce', 'T2', 'FLAIR']
MODALITY_PATTERNS = {
    'T1': '*t1n*', 'T1ce': '*t1c*', 'T2': '*t2w*', 'FLAIR': '*t2f*',
}
SEG_PATTERN = '*seg*'


def find_file(case_dir, pattern):
    matches = list(case_dir.glob(pattern + '.nii.gz')) + list(case_dir.glob(pattern + '.nii'))
    return matches[0] if matches else None


def load_mome_model(device, expert_dir):
    """Load MoMESegmenter with expert + fusion checkpoints."""
    model = MoMESegmenter(
        modalities=MODALITY_ORDER, in_channels=1, num_classes=3,
        base_channels=32, depth=4, attention_type='cbam', fusion_method='weighted',
    )
    for mod in MODALITY_ORDER:
        ckpt_path = expert_dir / f'expert_{mod}_best.pth'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.experts[mod].load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
            print(f"  Loaded expert: {mod}")
        else:
            print(f"  WARNING: Missing {ckpt_path}")

    fusion_path = expert_dir / 'mome_fusion_best.pth'
    if fusion_path.exists():
        f_ckpt = torch.load(fusion_path, map_location=device, weights_only=False)
        model.load_state_dict(f_ckpt.get('model_state_dict', f_ckpt), strict=False)
        print(f"  Loaded fusion/gating weights")

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MoME+ Inference + Visualization')
    parser.add_argument('--case_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='output/slice_visualizations')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--display_modality', type=str, default='T1ce',
                        choices=MODALITY_ORDER)
    parser.add_argument('--expert_dir', type=str,
                        default='experiments/checkpoints/experts')
    parser.add_argument('--with_gt', action='store_true',
                        help='Also visualize ground truth mask')
    args = parser.parse_args()

    case_dir   = Path(args.case_dir)
    output_dir = Path(args.output)
    expert_dir = Path(args.expert_dir)
    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else ('cpu' if args.device == 'auto' else args.device)
    )

    print(f"\n{'='*60}")
    print(f"MoME+ Inference + Visualization")
    print(f"  Case:   {case_dir.name}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # --- 1. Load MRIs ---
    print(f"\n[1/4] Loading MRI modalities...")
    modality_paths = {}
    for mod, pattern in MODALITY_PATTERNS.items():
        fpath = find_file(case_dir, pattern)
        if fpath:
            modality_paths[mod] = str(fpath)
            print(f"  {mod}: {fpath.name}")
        else:
            print(f"  {mod}: NOT FOUND")

    if not modality_paths:
        print("ERROR: No MRI files found!")
        return

    # --- 2. Load model + create InferenceEngine ---
    print(f"\n[2/4] Loading MoME+ model...")
    model = load_mome_model(device, expert_dir)

    engine = InferenceEngine(
        model=model, device=str(device),
        roi_size=(96, 96, 96), overlap=0.5, sw_batch_size=4,
        use_amp=(device.type == 'cuda'),
    )

    # --- 3. Load volumes and run inference using InferenceEngine ---
    print(f"\n[3/4] Running inference via InferenceEngine...")
    volumes = {}
    reference_nifti = None
    for mod, path in modality_paths.items():
        vol, nifti = engine.load_nifti(path)
        volumes[mod] = vol
        if reference_nifti is None:
            reference_nifti = nifti
        print(f"  Loaded {mod}: shape {vol.shape}")

    result = engine.predict_full_mome(volumes)
    brats_mask = result['segmentation']    # BraTS labels (0/1/2/4), same shape as input

    wt_count = (brats_mask > 0).sum()
    tc_count = ((brats_mask == 1) | (brats_mask == 4)).sum()
    et_count = (brats_mask == 4).sum()
    print(f"\n  Predicted mask shape: {brats_mask.shape}")
    print(f"  WT voxels: {wt_count:,}")
    print(f"  TC voxels: {tc_count:,}")
    print(f"  ET voxels: {et_count:,}")

    # Save predicted mask as NIfTI
    output_dir.mkdir(parents=True, exist_ok=True)
    engine.save_prediction(brats_mask, reference_nifti,
                           str(output_dir / f'{case_dir.name}_predicted_mask.nii.gz'))

    # --- 4. Visualize ---
    print(f"\n[4/4] Generating visualizations (display: {args.display_modality})...")
    viz = SliceVisualizer()

    display_mod = args.display_modality if args.display_modality in volumes else next(iter(volumes))
    display_vol = volumes[display_mod]
    case_name = case_dir.name

    # Prediction overlay
    results = viz.generate_from_arrays(
        mri_volume=display_vol, brats_mask=brats_mask,
        output_dir=output_dir,
        prefix=f'{case_name}_pred_overlay', overlay_mode=True,
    )
    print(f"\n  Prediction overlay:")
    for name, path in results.items():
        print(f"    {name}: {path}")

    # Prediction standalone
    results = viz.generate_from_arrays(
        mri_volume=display_vol, brats_mask=brats_mask,
        output_dir=output_dir,
        prefix=f'{case_name}_pred_standalone', overlay_mode=False,
    )
    print(f"\n  Prediction standalone:")
    for name, path in results.items():
        print(f"    {name}: {path}")

    # Ground truth comparison
    if args.with_gt:
        gt_path = find_file(case_dir, SEG_PATTERN)
        if gt_path:
            gt_nii = nib.load(str(gt_path))
            gt_mask = gt_nii.get_fdata().astype(np.uint8)

            results = viz.generate_from_arrays(
                mri_volume=display_vol, brats_mask=gt_mask,
                output_dir=output_dir,
                prefix=f'{case_name}_gt_overlay', overlay_mode=True,
            )
            print(f"\n  Ground truth overlay:")
            for name, path in results.items():
                print(f"    {name}: {path}")
        else:
            print("  No ground truth mask found.")

    print(f"\n{'='*60}")
    print(f"Done! Output: {output_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
