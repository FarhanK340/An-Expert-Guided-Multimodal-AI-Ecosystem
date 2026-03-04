"""
Train report generator with LoRA fine-tuning on Gemma-2B-Instruct.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report_generation.dataset import ReportDataset


def main():
    parser = argparse.ArgumentParser(description="Train report generation model")
    parser.add_argument('--train_data', type=str, default='data/reports/train.jsonl',
                        help='Training data path')
    parser.add_argument('--val_data', type=str, default='data/reports/val.jsonl',
                        help='Validation data path')
    parser.add_argument('--model_name', type=str, default='google/gemma-2b-it',
                        help='Base model name')
    parser.add_argument('--output_dir', type=str, default='models/report_generator',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per device')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Report Generator Training")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Training data: {args.train_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ReportDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = ReportDataset(args.val_data, tokenizer, args.max_length)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Load base model
    print("Loading base model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=device == "cuda",
        report_to="tensorboard",
        save_total_limit=3,
        remove_unused_columns=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    
    print("=" * 60)
    print("✅ Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
