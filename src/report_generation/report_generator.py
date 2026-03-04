"""
Report generator classes for inference.
"""

import torch
import json
from typing import Dict, Any, Optional
from pathlib import Path

from .templates import generate_template_report


class TemplateBasedGenerator:
    """Baseline template-based report generator."""
    
    def __init__(self):
        """Initialize template generator."""
        pass
    
    def generate(self, json_data: Dict[str, Any], template_idx: int = 0) -> str:
        """
        Generate report using deterministic templates.
        
        Args:
            json_data: Structured JSON anatomical descriptors
            template_idx: Which template variation to use (0-4)
        
        Returns:
            Generated report text
        """
        return generate_template_report(json_data, template_idx)
    
    def batch_generate(self, json_data_list: list, template_idx: int = 0) -> list:
        """Generate reports for a batch of JSON inputs."""
        return [self.generate(data, template_idx) for data in json_data_list]


class ReportGenerator:
    """LLM-based report generator with LoRA fine-tuning."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2b-it",
        lora_checkpoint: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize LLM report generator.
        
        Args:
            model_name: HuggingFace model name
            lora_checkpoint: Path to LoRA adapter checkpoint
            device: Device to run on (cuda/cpu)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Import here to avoid requiring transformers if not using this class
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Load LoRA adapter if provided
        if lora_checkpoint:
            print(f"Loading LoRA adapter from: {lora_checkpoint}")
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)
            self.model = self.model.merge_and_unload()  # Merge for faster inference
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def generate(
        self,
        json_data: Dict[str, Any],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate report from JSON data using LLM.
        
        Args:
            json_data: Structured JSON anatomical descriptors
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated report text
        """
        # Format JSON
        json_str = json.dumps(json_data, indent=2)
        
        # Create prompt
        prompt = (
            "Generate a professional radiology report from the following "
            "structured medical imaging data:\n\n"
            f"{json_str}\n\n"
            "Report:\n"  # Added \n to match training format!
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after prompt)
        if "Report:" in generated_text:
            report = generated_text.split("Report:")[-1].strip()
        else:
            report = generated_text[len(prompt):].strip()
        
        return report
    
    def batch_generate(
        self,
        json_data_list: list,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> list:
        """Generate reports for a batch of JSON inputs."""
        reports = []
        for json_data in json_data_list:
            report = self.generate(json_data, max_new_tokens, temperature, top_p)
            reports.append(report)
        return reports
