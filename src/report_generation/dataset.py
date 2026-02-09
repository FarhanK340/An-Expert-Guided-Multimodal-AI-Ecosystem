"""
PyTorch dataset for JSON-to-report pairs.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from pathlib import Path


class ReportDataset(Dataset):
    """Dataset for training report generation models."""
    
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int = 512,
        prompt_template: Optional[str] = None
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with training data
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            prompt_template: Optional custom prompt template
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Default instruction-tuned prompt
        if prompt_template is None:
            self.prompt_template = (
                "Generate a professional radiology report from the following "
                "structured medical imaging data:\n\n{json_str}\n\n"
                "Report:"
            )
        else:
            self.prompt_template = prompt_template
        
        # Load data
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract JSON and report
        json_data = item['json_data']
        report = item['report']
        
        # Format JSON for readability
        json_str = json.dumps(json_data, indent=2)
        
        # Create prompt
        prompt = self.prompt_template.format(json_str=json_str)
        
        # Full text for training (prompt + report)
        full_text = prompt + "\n" + report
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids, but with prompt masked)
        # For instruction tuning, we only compute loss on the report part
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True
        )
        prompt_length = len(prompt_encoding['input_ids'])
        
        labels = encoding['input_ids'].clone()
        labels[:, :prompt_length] = -100  # Mask prompt tokens
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """Save data to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
