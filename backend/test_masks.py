from cases.models import SegmentationResult
from django.conf import settings
import pathlib

segs = list(SegmentationResult.objects.all())
print(f"Total segs: {len(segs)}")
for seg in segs:
    print(f"\n--- Case: {seg.case.case_id} ---")
    struct = seg.structured_findings or {}
    mask = struct.get('full_segmentation_mask')
    print("Full mask in struct:", mask)
    if mask:
        print("Mask exists on disk?", (pathlib.Path(settings.MEDIA_ROOT) / str(mask)).exists())
    
    # Try the relpath block
    import os
    if mask:
        rel = os.path.relpath(str(mask), settings.MEDIA_ROOT)
        rel = rel.replace('\\', '/')
        print("Rel URL component:", rel)
