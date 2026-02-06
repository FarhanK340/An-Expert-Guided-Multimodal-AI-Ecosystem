"""
Wrapper to fix Python path and run train_single_expert.py
This ensures venv packages are loaded before conda packages.
"""
import sys
import os

# Get the venv site-packages directory
venv_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
venv_site_packages = os.path.join(venv_root, '.venv', 'Lib', 'site-packages')

# Remove conda from sys.path
sys.path = [p for p in sys.path if 'miniconda3' not in p.lower()]

# Insert venv site-packages at the beginning (position 1, after current dir)
if venv_site_packages not in sys.path:
    sys.path.insert(1, venv_site_packages)

print(f"✓ Fixed Python path. Venv packages will load first.")
print(f"  Venv: {venv_site_packages}")

# Now run the actual training script
exec(open(os.path.join(venv_root, 'src', 'training', 'train_single_expert.py')).read())
