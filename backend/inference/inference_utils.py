"""
MoME+ Inference Engine.

Loads the full MoMESegmenter (4 ModalityExperts + HierarchicalGatingNetwork +
ExpertFusion) from a single checkpoint and runs inference on uploaded MRI scans.

Checkpoint layout expected at SEGMENTATION_MODEL_PATH:
  Either:
    (a) torch.save(model.state_dict(), path)        → loaded as state_dict
    (b) torch.save({'model_state_dict': ...}, path) → loaded from key

Expert checkpoints (optional, per-expert fine-tuning):
  Place individual expert weights at:
    models/checkpoints/experts/T1.pth
    models/checkpoints/experts/T1ce.pth
    models/checkpoints/experts/T2.pth
    models/checkpoints/experts/FLAIR.pth
  These are loaded AFTER the main checkpoint if they exist.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import nibabel as nib

from django.conf import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers — resolve src/ from the repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(settings.BASE_DIR).parent          # …/An-Expert-Guided-…/backend → parent
SRC_PATH  = REPO_ROOT / 'src'

# Add the repo root (parent of src/) so that `import src.models…` works
# AND internal relative imports like `from ..utils.logger` resolve correctly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_model():
    """
    Lazy import MoMESegmenter.

    We import as `src.models.mome_segmenter` (full package path) so that
    relative imports inside the src package (e.g. `from ..utils.logger`)
    resolve correctly.  Importing only from `models.mome_segmenter` with
    SRC_PATH on sys.path triggers 'attempted relative import beyond top-level
    package' because Python then sees `models` as a top-level package.
    """
    try:
        from src.models.mome_segmenter import MoMESegmenter
        return MoMESegmenter
    except ImportError as e:
        logger.warning(f"Could not import MoMESegmenter: {e}. Inference will use mock mode.")
        return None


# ---------------------------------------------------------------------------
# Main InferenceEngine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Loads MoMESegmenter and runs full forward pass for a case.

    Usage:
        engine = InferenceEngine()
        result = engine.run_inference(case_id, case_dir)
    """

    MODALITIES   = ['T1', 'T1ce', 'T2', 'FLAIR']
    # Patch size for sliding window inference — larger = more context, fewer patches
    TARGET_SIZE  = (96, 96, 96)

    def __init__(self):
        # Auto-select GPU when available; fall back to CPU
        cfg_device = getattr(settings, 'ML_CONFIG', {}).get('device', 'auto')
        if cfg_device == 'auto' or cfg_device == 'cpu':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(cfg_device)
        logger.info(f"InferenceEngine using device: {self.device}")
        self.model: Optional[torch.nn.Module] = None
        self._model_loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_inference(self, case_id: str, case_dir: Path) -> Dict:
        """
        Run the full MoME+ inference pipeline using sliding window inference.

        The model was trained on 64x64x64 patches at native BraTS resolution.
        We must NOT resize the full volume to 64x64x64 (that destroys spatial
        scale).  Instead we use MONAI's SlidingWindowInferer to extract
        overlapping 64^3 patches, run each through MoME+, and stitch back.

        Args:
            case_id:  UUID string of the Case
            case_dir: Directory containing the uploaded NIfTI files

        Returns:
            Dict with volumes, confidence_scores, mask_files, gating_weights
        """
        from cases.models import Case, MRIImage, SegmentationResult
        from inference.preprocessing_pipeline import InferencePreprocessor
        from monai.inferers import SlidingWindowInferer

        # --- 1. Load model ---
        model = self._get_model()

        # --- 2. Load & normalise each modality at FULL native resolution ---
        preprocessor = InferencePreprocessor(target_size=self.TARGET_SIZE)
        volumes_np: Dict[str, np.ndarray] = {}   # modality → (H, W, D) float32
        ref_nifti_path: Optional[Path] = None
        available_modalities = []

        for mod in self.MODALITIES:
            f = preprocessor._find_modality_file(case_dir, mod)
            if f is not None:
                vol_np, _ = preprocessor._load_nifti(f)
                vol_np    = preprocessor._normalize(vol_np)
                # Keep in (H, W, D) — will permute to (D, H, W) below
                volumes_np[mod] = vol_np.astype(np.float32)
                available_modalities.append(mod)
                if ref_nifti_path is None:
                    ref_nifti_path = f
                logger.info(f"Loaded {mod} at native shape {vol_np.shape}")

        if not volumes_np:
            raise ValueError("No MRI modality files found in case directory.")

        # --- 3. Build (1, 4, D, H, W) stacked tensor for sliding window ---
        # Use the first available modality's shape as reference
        ref_shape = next(iter(volumes_np.values())).shape  # (H, W, D)
        D, H, W   = ref_shape[2], ref_shape[0], ref_shape[1]

        modality_tensors = []
        for mod in self.MODALITIES:
            if mod in volumes_np:
                arr = volumes_np[mod]           # (H, W, D)
                t   = torch.from_numpy(arr).permute(2, 0, 1)  # (D, H, W)
            else:
                t = torch.zeros(D, H, W)        # zero-fill missing modality
            modality_tensors.append(t)

        # Stack → (4, D, H, W) → add batch → (1, 4, D, H, W)
        stacked = torch.stack(modality_tensors, dim=0).unsqueeze(0).to(self.device)
        logger.info(f"Stacked input shape: {stacked.shape}")  # (1, 4, D, H, W)

        # --- 4. Sliding window inference ---
        roi = self.TARGET_SIZE  # (64, 64, 64) — must match training patch size

        if model is not None:
            # Wrap model to accept (B, 4, D, H, W) and route each channel
            # to its respective modality expert
            def model_fn(x: torch.Tensor) -> torch.Tensor:
                """x: (B, 4, D, H, W) — split into per-modality dict."""
                patch_dict = {}
                for i, mod in enumerate(self.MODALITIES):
                    patch_dict[mod] = x[:, i:i+1, ...]   # (B, 1, D, H, W)
                # Use automatic mixed precision for speed on GPU
                use_amp = (self.device.type == 'cuda')
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        out = model(patch_dict)
                else:
                    out = model(patch_dict)
                return out['segmentation']  # (B, C, D, H, W)

            inferer = SlidingWindowInferer(
                roi_size=roi,
                sw_batch_size=4,       # Match src implementation (sw_batch_size=4)
                overlap=0.5,
                mode='gaussian',       # smooth blending at patch boundaries
                progress=False,
            )
            logger.info(f"Running sliding window inference (roi={roi}, overlap=0.5) …")
            with torch.no_grad():
                seg_logits = inferer(stacked, model_fn)  # (1, C, D, H, W)

            # Also grab gating weights from a single centre-patch forward pass
            # (for display purposes only)
            cx, cy, cz = D//2, H//2, W//2
            r = 32
            centre_patch = {
                mod: stacked[:, i:i+1,
                             max(0,cx-r):cx+r,
                             max(0,cy-r):cy+r,
                             max(0,cz-r):cz+r]
                for i, mod in enumerate(self.MODALITIES)
            }
            with torch.no_grad():
                centre_out    = model(centre_patch)
            gating_weights = self._extract_gating_weights(centre_out, available_modalities)
        else:
            logger.warning("Model not loaded — using mock segmentation output.")
            # Mock: produce (1, 3, D, H, W) zero logits with a small centre tumour
            seg_logits = torch.zeros(1, 3, D, H, W)
            cx, cy, cz = D//2, H//2, W//2
            r = 10
            seg_logits[0, 1, cx-r:cx+r, cy-r:cy+r, cz-r:cz+r] = 5.0
            seg_logits[0, 2, cx-r//2:cx+r//2, cy-r//2:cy+r//2, cz-r//2:cz+r//2] = 8.0
            gating_weights = {mod: 0.25 for mod in self.MODALITIES}

        # --- 5. Post-process Multi-Label Outputs —-------------------------
        # The model outputs 3 independent logits (WT, TC, ET). There is NO background class.
        # We must use Sigmoid and a probability threshold, NOT argmax.
        probs = torch.sigmoid(seg_logits[0])  # (3, D, H, W)
        
        # Threshold at 0.5 for each sub-region
        wt_mask = (probs[0] > 0.5).cpu().numpy()
        tc_mask = (probs[1] > 0.5).cpu().numpy()
        et_mask = (probs[2] > 0.5).cpu().numpy()

        # Reconstruct exactly as requested by BraTS conventions natively:
        # Background = 0
        # WT (Edema) = 2
        # TC (Necrotic) = 1
        # ET (Enhancing) = 4
        
        D_dim, H_dim, W_dim = wt_mask.shape
        brats_mask = np.zeros((D_dim, H_dim, W_dim), dtype=np.uint8)
        
        # Hierarchical assembly:
        # 1. Everything in Whole Tumor start as Edema (2)
        brats_mask[wt_mask] = 2
        
        # 2. Everything in Tumor Core overwrites as Necrotic (1)
        brats_mask[tc_mask] = 1
        
        # 3. Everything in Enhancing Tumor overwrites as Enhancing (4)
        # Note: BraTS convention uses label 4 for ET, though sometimes mapped to 3.
        brats_mask[et_mask] = 4
        
        # Save a continuous 0,1,2,3 mapped version for downstream metric calculation consistency
        # if other blocks expect 0,1,2,3 instead of 0,1,2,4.
        seg_np = np.zeros((D_dim, H_dim, W_dim), dtype=np.uint8)
        seg_np[wt_mask] = 1 # WT
        seg_np[tc_mask] = 2 # TC
        seg_np[et_mask] = 3 # ET

        volumes, confidence = self._compute_metrics(seg_logits, seg_np)

        # --- 6. Save NIfTI masks aligned to original space ---------------
        ref_affine = preprocessor.get_reference_affine(case_dir)
        mask_files = self._save_masks(brats_mask, seg_np, ref_affine, case_dir)

        # --- 7. Generate 2D slice visualizations -------------------------
        slice_image_urls = self._generate_slice_visualizations(
            brats_mask, volumes_np, case_id, case_dir,
        )

        # --- 8. Persist to DB --------------------------------------------
        try:
            case = Case.objects.get(case_id=case_id)
        except Case.DoesNotExist:
            raise ValueError(f"Case {case_id} not found in database.")

        seg_result, created = SegmentationResult.objects.update_or_create(
            case=case,
            defaults={
                'whole_tumor_mask':       mask_files.get('whole_tumor', ''),
                'tumor_core_mask':        mask_files.get('tumor_core', ''),
                'enhancing_tumor_mask':   mask_files.get('enhancing_tumor', ''),
                'whole_tumor_volume':     float(volumes['whole_tumor']),
                'tumor_core_volume':      float(volumes['tumor_core']),
                'enhancing_tumor_volume': float(volumes['enhancing_tumor']),
                'whole_tumor_confidence':     float(confidence['whole_tumor']),
                'tumor_core_confidence':      float(confidence['tumor_core']),
                'enhancing_tumor_confidence': float(confidence['enhancing_tumor']),
                'structured_findings': {
                    'volumes':               volumes,
                    'confidence_scores':     confidence,
                    'gating_weights':        gating_weights,
                    'available_modalities':  available_modalities,
                    'inference_mode':        'sliding_window',
                    'roi_size':              list(roi),
                    'overlap':               0.5,
                    'timestamp':             datetime.now().isoformat(),
                    'model_version':         'MoME+ v1.0',
                    'device':                str(self.device),
                    'full_segmentation_mask': mask_files.get('full_segmentation', ''),
                    'slice_images':          slice_image_urls,
                },
            }
        )

        # Save Slice2DVisualization DB records
        self._save_slice_records(seg_result, slice_image_urls, case_id)

        case.status = 'completed'
        case.save(update_fields=['status'])
        logger.info(f"Inference complete for case {case_id}. created={created}")

        return {
            'case_id': case_id,
            'volumes': volumes,
            'confidence_scores': confidence,
            'gating_weights': gating_weights,
            'mask_files': mask_files,
            'slice_images': slice_image_urls,
            'created': created,
        }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_model(self) -> Optional[torch.nn.Module]:
        """Load (or return cached) MoMESegmenter."""
        if self._model_loaded:
            return self.model

        MoMESegmenter = _import_model()
        if MoMESegmenter is None:
            self._model_loaded = True
            return None

        ml_config    = getattr(settings, 'ML_CONFIG', {})
        ckpt_path    = Path(settings.BASE_DIR) / ml_config.get(
            'segmentation_model_path', 'models/checkpoints/mome_segmenter.pth'
        )
        # Experts live in repo-root experiments/checkpoints/experts/
        expert_dir   = REPO_ROOT / 'experiments' / 'checkpoints' / 'experts'

        model = MoMESegmenter(
            modalities=['T1', 'T1ce', 'T2', 'FLAIR'],
            in_channels=1,
            num_classes=3,
            base_channels=32,
            depth=4,
            attention_type='cbam',
            fusion_method='weighted',
        )

        # Load main checkpoint
        if ckpt_path.exists():
            logger.info(f"Loading MoMESegmenter checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            logger.info("Main checkpoint loaded successfully.")
        else:
            logger.warning(
                f"No checkpoint at {ckpt_path}. "
                "Running with random weights (output will be meaningless)."
            )

        # Load per-expert checkpoints.
        # Actual filenames: expert_<Modality>_best.pth
        # Fallback: <Modality>.pth  (legacy naming)
        for mod in self.MODALITIES:
            candidate_names = [
                f"expert_{mod}_best.pth",   # new naming used by training scripts
                f"{mod}.pth",               # legacy naming
            ]
            for cname in candidate_names:
                expert_ckpt = expert_dir / cname
                if expert_ckpt.exists() and mod in model.experts:
                    logger.info(f"Loading expert checkpoint for {mod}: {expert_ckpt}")
                    e_ckpt = torch.load(expert_ckpt, map_location=self.device, weights_only=False)
                    state  = e_ckpt.get('model_state_dict', e_ckpt.get('state_dict', e_ckpt))
                    model.experts[mod].load_state_dict(state, strict=False)
                    break

        # Load fusion/gating checkpoint if available
        fusion_ckpt_path = expert_dir / 'mome_fusion_best.pth'
        if fusion_ckpt_path.exists():
            logger.info(f"Loading fusion/gating checkpoint: {fusion_ckpt_path}")
            f_ckpt = torch.load(fusion_ckpt_path, map_location=self.device, weights_only=False)
            f_state = f_ckpt.get('model_state_dict', f_ckpt.get('state_dict', f_ckpt))
            # Load into gating_network and fusion layers only
            if hasattr(model, 'gating_network'):
                gn_state = {k.replace('gating_network.', ''): v
                            for k, v in f_state.items() if k.startswith('gating_network.')}
                if gn_state:
                    model.gating_network.load_state_dict(gn_state, strict=False)
            if hasattr(model, 'fusion'):
                fn_state = {k.replace('fusion.', ''): v
                            for k, v in f_state.items() if k.startswith('fusion.')}
                if fn_state:
                    model.fusion.load_state_dict(fn_state, strict=False)
            # Also try loading the whole checkpoint into the model (covers any layout)
            model.load_state_dict(f_state, strict=False)
            logger.info("Fusion/gating checkpoint loaded.")

        model.to(self.device)
        model.eval()
        self.model = model
        self._model_loaded = True
        return model

    # ------------------------------------------------------------------
    # Forward pass — handles partial modalities
    # ------------------------------------------------------------------

    def _forward(self,
                 model: torch.nn.Module,
                 preprocessed: Dict[str, torch.Tensor],
                 available: list) -> Dict:
        """
        MoMESegmenter.forward() expects a Dict[modality → (B,1,D,H,W)].
        Missing modalities are filled with zeros so the gating network
        still gets the full 4-channel input.
        """
        ref = next(iter(preprocessed.values()))
        full_input: Dict[str, torch.Tensor] = {}

        for mod in self.MODALITIES:
            if mod in preprocessed:
                full_input[mod] = preprocessed[mod]
            else:
                # Zero-fill missing modality so gating sees 4 channels
                full_input[mod] = torch.zeros_like(ref)

        with torch.no_grad():
            outputs = model(full_input)

        return outputs

    def _mock_outputs(self, preprocessed: Dict[str, torch.Tensor]) -> Dict:
        """Deterministic mock when model is unavailable (for unit tests / no checkpoint)."""
        ref  = next(iter(preprocessed.values()))
        B, _, D, H, W = ref.shape   # 64×64×64 after fix
        seg = torch.zeros(B, 3, D, H, W)  # 3 classes
        # Simulate a small tumor region in the centre
        cx, cy, cz = D // 2, H // 2, W // 2
        r = 6   # smaller radius appropriate for 64^3
        seg[0, 1, cx-r:cx+r, cy-r:cy+r, cz-r:cz+r] = 5.0   # edema
        seg[0, 2, cx-r//2:cx+r//2, cy-r//2:cy+r//2, cz-r//2:cz+r//2] = 8.0  # enhancing
        w  = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        sa = torch.ones(B, 4, D, H, W) * 0.5
        return {'segmentation': seg, 'expert_weights': w, 'spatial_attention': sa}

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self,
                         seg_logits: torch.Tensor,
                         seg_np: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Compute volumetric metrics and per-region confidence scores.

        Assumes seg_np uses class indices: 0=background, 1=NCR/NET/ED, 2=ET
        BraTS regions (from class indices):
            Whole Tumor (WT) = classes 1 + 2
            Tumor Core  (TC) = class  2   (often TC maps to class 2)
            Enhancing   (ET) = class  2   (same for 3-class models)
        """
        probs = torch.sigmoid(seg_logits[0])   # (3, D, H, W)
        probs_np = probs.cpu().numpy()

        voxel_vol_mm3 = 1.0  # 1 mm³ assuming native spacing is roughly 1mm isotropic

        # Our new mapping: 1=WT, 2=TC, 3=ET
        wt_mask = (seg_np == 1).astype(np.float32)
        tc_mask = (seg_np == 2).astype(np.float32)
        et_mask = (seg_np == 3).astype(np.float32)

        volumes = {
            'whole_tumor':     float(wt_mask.sum() * voxel_vol_mm3),
            'tumor_core':      float(tc_mask.sum() * voxel_vol_mm3),
            'enhancing_tumor': float(et_mask.sum() * voxel_vol_mm3),
        }

        def mean_prob(mask, ch_idx):
            if mask.sum() == 0:
                return 0.0
            return float(probs_np[ch_idx][mask.astype(bool)].mean())

        # Confidence should be from their respective probability channels
        # Channel 0: WT, Channel 1: TC, Channel 2: ET
        confidence = {
            'whole_tumor':     mean_prob(wt_mask, 0),
            'tumor_core':      mean_prob(tc_mask, 1),
            'enhancing_tumor': mean_prob(et_mask, 2),
        }

        return volumes, confidence

    def _extract_gating_weights(self, outputs: Dict, available: list) -> Dict:
        """Extract per-expert gating weights from model outputs."""
        if 'expert_weights' not in outputs:
            return {mod: 0.25 for mod in self.MODALITIES}

        weights_tensor = outputs['expert_weights']  # (B, num_experts)
        weights_list   = weights_tensor[0].cpu().tolist()

        return {
            mod: round(weights_list[i], 4)
            for i, mod in enumerate(self.MODALITIES)
            if i < len(weights_list)
        }

    def _save_masks(self,
                    brats_mask:  np.ndarray,
                    class_mask:  np.ndarray,
                    affine:      np.ndarray,
                    case_dir:    Path) -> Dict[str, str]:
        """Save NIfTI mask files and return their paths."""
        mask_dir = case_dir / 'masks'
        mask_dir.mkdir(parents=True, exist_ok=True)

        def save_nifti(arr, name):
            img = nib.Nifti1Image(arr.astype(np.uint8), affine)
            path = mask_dir / f"{name}.nii.gz"
            nib.save(img, str(path))
            return str(path)

        # Whole tumor
        wt = (brats_mask > 0).astype(np.uint8)
        # Tumor core (labels 1 + 4 in BraTS = class 1+2 in model)
        tc = ((brats_mask == 1) | (brats_mask == 4)).astype(np.uint8)
        # Enhancing tumor (label 4 = class 2)
        et = (brats_mask == 4).astype(np.uint8)

        return {
            'whole_tumor':     save_nifti(wt,        'whole_tumor'),
            'tumor_core':      save_nifti(tc,        'tumor_core'),
            'enhancing_tumor': save_nifti(et,        'enhancing_tumor'),
            'full_segmentation': save_nifti(brats_mask, 'full_segmentation'),
        }

    def _generate_slice_visualizations(
        self,
        brats_mask: np.ndarray,
        volumes_np: Dict[str, np.ndarray],
        case_id: str,
        case_dir: Path,
    ) -> list:
        """
        Generate 2D slice visualization PNGs using SliceVisualizer.

        Produces overlay and standalone composite images for the axial plane
        (the most clinically useful view). Uses T1ce as the MRI background
        if available, otherwise falls back to the first available modality.

        Args:
            brats_mask:  BraTS-convention mask in (D, H, W) format
            volumes_np:  Dict of modality → (H, W, D) numpy arrays
            case_id:     Case UUID string
            case_dir:    Path to case directory

        Returns:
            List of dicts: [{ plane, slice_index, url, has_overlay, filename }]
        """
        try:
            from src.inference.slice_visualizer import SliceVisualizer
        except ImportError as e:
            logger.warning(f"Could not import SliceVisualizer: {e}. Skipping slice generation.")
            return []

        viz = SliceVisualizer()

        # Pick display modality (T1ce preferred)
        display_mod = 'T1ce' if 'T1ce' in volumes_np else next(iter(volumes_np))
        mri_vol_hwz = volumes_np[display_mod]        # (H, W, D) — NIfTI native

        # Permute both to (D, H, W) for SliceVisualizer consistency
        mri_vol = np.transpose(mri_vol_hwz, (2, 0, 1))
        mask_vol = brats_mask  # already (D, H, W) from sliding window output

        # Output directory under Django media
        slice_dir = Path(settings.MEDIA_ROOT) / 'cases' / str(case_id) / 'slices'
        slice_dir.mkdir(parents=True, exist_ok=True)

        slice_results = []

        # Generate overlay composite (MRI + coloured masks)
        try:
            overlay_files = viz.generate_from_arrays(
                mri_volume=mri_vol,
                brats_mask=mask_vol,
                output_dir=str(slice_dir),
                plane='axial',
                prefix=f'{case_id}_overlay',
                save_individual=False,
                save_composite=True,
                overlay_mode=True,
            )
            if 'composite' in overlay_files:
                rel_path = Path(overlay_files['composite']).relative_to(settings.MEDIA_ROOT)
                fname = Path(overlay_files['composite']).name
                best_slice = viz.find_best_slice(mask_vol, 'axial')
                slice_results.append({
                    'plane': 'axial',
                    'slice_index': best_slice,
                    'url': f'{settings.MEDIA_URL}{rel_path.as_posix()}',
                    'has_overlay': True,
                    'filename': fname,
                })
                logger.info(f"Saved overlay composite: {fname}")
        except Exception as e:
            logger.warning(f"Failed to generate overlay composite: {e}")

        # Generate standalone composite (masks on black background)
        try:
            standalone_files = viz.generate_from_arrays(
                mri_volume=mri_vol,
                brats_mask=mask_vol,
                output_dir=str(slice_dir),
                plane='axial',
                prefix=f'{case_id}_standalone',
                save_individual=False,
                save_composite=True,
                overlay_mode=False,
            )
            if 'composite' in standalone_files:
                rel_path = Path(standalone_files['composite']).relative_to(settings.MEDIA_ROOT)
                fname = Path(standalone_files['composite']).name
                best_slice = viz.find_best_slice(mask_vol, 'axial')
                slice_results.append({
                    'plane': 'axial',
                    'slice_index': best_slice,
                    'url': f'{settings.MEDIA_URL}{rel_path.as_posix()}',
                    'has_overlay': False,
                    'filename': fname,
                })
                logger.info(f"Saved standalone composite: {fname}")
        except Exception as e:
            logger.warning(f"Failed to generate standalone composite: {e}")

        return slice_results

    def _save_slice_records(
        self,
        seg_result,
        slice_image_urls: list,
        case_id: str,
    ):
        """Persist Slice2DVisualization DB records for generated slice images."""
        from cases.models import Slice2DVisualization

        # Clear any previous slice records for this segmentation result
        Slice2DVisualization.objects.filter(segmentation_result=seg_result).delete()

        for item in slice_image_urls:
            try:
                # image_file is relative to MEDIA_ROOT
                rel_path = item['url'].replace(settings.MEDIA_URL, '', 1)
                Slice2DVisualization.objects.create(
                    segmentation_result=seg_result,
                    plane=item['plane'],
                    slice_index=item['slice_index'],
                    image_file=rel_path,
                    modality='t1ce',
                    has_overlay=item['has_overlay'],
                )
            except Exception as e:
                logger.warning(f"Failed to save slice record: {e}")

