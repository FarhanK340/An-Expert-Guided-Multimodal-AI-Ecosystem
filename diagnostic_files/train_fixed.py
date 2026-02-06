"""
Fix Python path to prioritize venv over conda, then run train_single_expert.py
"""
import sys
import os

# Remove conda from sys.path BEFORE any imports
original_path = sys.path.copy()
sys.path = [p for p in sys.path if 'miniconda3' not in p.lower() and 'conda' not in p.lower()]

print(f"Removed conda from Python path")
print(f"  Before: {len(original_path)} paths")
print(f"  After: {len(sys.path)} paths")
print()

# Ensure project root is in path for 'src' imports
# Ensure project root is in path for 'src' imports
# Since this file is in 'diagnostic_files', we need to go up one level to find the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and run the training script's main function
# We need to use runpy to preserve __name__ == '__main__'
import runpy
sys.argv = sys.argv  # Keep original arguments
runpy.run_path(os.path.join(project_root, 'src', 'training', 'train_single_expert.py'), run_name='__main__')
