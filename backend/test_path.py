import os

path_str = r"C:\Users\Farhan\Desktop\FYP\An-Expert-Guided-Multimodal-AI-Ecosystem\backend\media\cases\x\full_segmentation.nii.gz"
media_root = r"c:\Users\Farhan\Desktop\FYP\An-Expert-Guided-Multimodal-AI-Ecosystem\backend\media"

def mask_url(field_value):
    if not field_value: return None
    p = str(field_value).replace('\\', '/')
    mr = str(media_root).replace('\\', '/')
    
    if p.lower().startswith(mr.lower()):
        rel = p[len(mr):].lstrip('/')
        return f"/media/{rel}"
        
    return p

print("URL:", mask_url(path_str))
