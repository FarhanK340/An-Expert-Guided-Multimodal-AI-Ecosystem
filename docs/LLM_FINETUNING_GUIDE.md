# MedGemma-4B Fine-Tuning for Medical Report Generation

## Overview

This guide provides a **production-grade pipeline** for fine-tuning **MedGemma-4B** (Google's medical LLM) on synthetic JSON→Report pairs for anatomically-aware radiology report generation.

**Key Features:**
- Instruction-tuned for medical domain
- LoRA/PEFT for efficient training on consumer GPUs
- Factual consistency constraints
- Safety guardrails for medical text
- Continual learning from clinician feedback

---

## Architecture Decision: Fine-Tuning vs RAG

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Full Fine-Tuning** | Best performance, no external dependencies | Expensive (40GB+ VRAM), slow | ❌ Not feasible for 8GB GPU |
| **LoRA/PEFT** | Efficient (8GB VRAM), fast inference, good performance | Slight quality trade-off | ✅ **RECOMMENDED** |
| **RAG** | No training, interpretable, easy updates | Retrieval overhead, context limits | ⚠️ Use as fallback |

**Decision: Use LoRA (Low-Rank Adaptation)**

---

## Stage 1: Environment Setup

### 1.1 Install Dependencies

```bash
pip install torch transformers accelerate peft datasets bitsandbytes \
    einops sentencepiece protobuf wandb
```

### 1.2 Model Access

MedGemma requires Hugging Face access:

```python
# Login to HuggingFace
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")

# Download MedGemma-4B
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/medgemma-4b"  # or medgemma-7b

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Quantization for 8GB GPU
    device_map='auto'
)
```

---

## Stage 2: Data Preparation

### 2.1 Format Conversion (JSONL → Instruction Format)

MedGemma expects instruction-formatted inputs:

```python
import json
from pathlib import Path

class MedGemmaDatasetFormatter:
    """
    Convert JSON→Report pairs to MedGemma instruction format.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Instruction template
        self.system_prompt = (
            "You are a board-certified radiologist. Generate a structured "
            "radiology report based on the provided brain tumor segmentation analysis. "
            "Be factual, concise, and use standard medical terminology."
        )
    
    def format_descriptor_as_prompt(self, descriptor):
        """
        Convert JSON descriptor to natural language prompt.
        
        Args:
            descriptor: Dictionary from synthetic dataset
        
        Returns:
            str: Formatted prompt
        """
        # Extract key information
        patient = descriptor['patient_info']
        seg = descriptor['segmentation_results']
        anatomy = descriptor['anatomical_mapping']
        
        # Build structured prompt
        prompt_parts = [
            f"Patient: {patient['age']}y {patient['sex']}, Case ID: {patient['case_id']}",
            f"Scan Date: {patient['scan_date']}",
            "",
            "Segmentation Analysis:",
            f"- Total Tumor Volume: {seg['volumetric_analysis']['total_tumor_volume_mm3']:.1f} mm³",
        ]
        
        # Tumor components
        components = seg['tumor_components']
        if components['enhancing_tumor']['present']:
            vol = components['enhancing_tumor']['volume_mm3']
            prompt_parts.append(f"- Enhancing Tumor: {vol:.1f} mm³")
        
        if components['necrotic_core']['present']:
            vol = components['necrotic_core']['volume_mm3']
            prompt_parts.append(f"- Necrotic Core: {vol:.1f} mm³")
        
        if components['peritumoral_edema']['present']:
            vol = components['peritumoral_edema']['volume_mm3']
            prompt_parts.append(f"- Peritumoral Edema: {vol:.1f} mm³")
        
        # Anatomical regions (top 5)
        prompt_parts.append("")
        prompt_parts.append("Affected Brain Regions:")
        for region in anatomy['affected_regions'][:5]:
            prompt_parts.append(
                f"- {region['region_name']} ({region.get('hemisphere', 'N/A')}): "
                f"{region['percentage_involvement']:.1f}% involvement"
            )
        
        # Clinical features
        clinical = descriptor.get('clinical_features', {})
        if clinical:
            prompt_parts.append("")
            prompt_parts.append("Clinical Features:")
            if clinical.get('mass_effect'):
                prompt_parts.append("- Mass effect present")
            if clinical.get('crossing_midline'):
                prompt_parts.append("- Crosses midline")
            if clinical.get('eloquent_area_involvement'):
                areas = ', '.join(clinical['eloquent_area_involvement'])
                prompt_parts.append(f"- Eloquent areas: {areas}")
        
        return '\n'.join(prompt_parts)
    
    def create_training_example(self, descriptor, report):
        """
        Format as instruction-following example.
        
        Returns:
            Dictionary with 'prompt' and 'completion' keys
        """
        # Build instruction prompt
        user_prompt = self.format_descriptor_as_prompt(descriptor)
        
        # MedGemma instruction format
        full_prompt = f"""<start_of_turn>system
{self.system_prompt}<end_of_turn>
<start_of_turn>user
Generate a radiology report based on the following brain tumor analysis:

{user_prompt}
<end_of_turn>
<start_of_turn>model
{report}<end_of_turn>"""
        
        return {
            'text': full_prompt,
            'descriptor': descriptor,  # Keep for validation
            'case_id': descriptor['patient_info']['case_id']
        }
    
    def process_dataset(self, input_jsonl, output_jsonl):
        """
        Convert entire dataset to instruction format.
        """
        examples = []
        
        with open(input_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line)
                example = self.create_training_example(
                    data['descriptor'],
                    data['report']
                )
                examples.append(example)
        
        # Save as JSONL
        with open(output_jsonl, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"✓ Processed {len(examples)} training examples")
        return examples
```

### 2.2 Load and Split Dataset

```python
from datasets import load_dataset

def prepare_training_data(formatted_jsonl, test_split=0.1):
    """
    Load and split dataset for training.
    
    Returns:
        (train_dataset, eval_dataset)
    """
    # Load from JSONL
    dataset = load_dataset('json', data_files=formatted_jsonl, split='train')
    
    # Train/test split
    split_dataset = dataset.train_test_split(test_size=test_split, seed=42)
    
    print(f"Training samples: {len(split_dataset['train'])}")
    print(f"Validation samples: {len(split_dataset['test'])}")
    
    return split_dataset['train'], split_dataset['test']
```

---

## Stage 3: LoRA Fine-Tuning Configuration

### 3.1 PEFT Configuration

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def setup_lora_model(base_model, tokenizer):
    """
    Configure LoRA for efficient fine-tuning.
    
    Returns:
        PEFT model ready for training
    """
    # Prepare model for k-bit training (8-bit quantization)
    model = prepare_model_for_kbit_training(base_model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,                      # LoRA rank (low-rank dimension)
        lora_alpha=32,             # Scaling factor
        target_modules=[           # Which layers to apply LoRA
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    # Expected: ~0.3-1% of total parameters
    
    return model
```

### 3.2 Training Arguments

```python
from transformers import TrainingArguments

def get_training_args(output_dir='./medgemma_finetuned'):
    """
    Optimized for 8GB GPU with gradient accumulation.
    """
    return TrainingArguments(
        # Output
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training hyperparameters
        num_train_epochs=3,
        per_device_train_batch_size=1,      # Small batch for 8GB GPU
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,      # Effective batch size = 8
        
        # Learning rate
        learning_rate=2e-4,
        lr_scheduler_type='cosine',
        warmup_steps=100,
        
        # Optimization
        optim='paged_adamw_8bit',           # Memory-efficient optimizer
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Memory optimization
        gradient_checkpointing=True,
        fp16=True,                          # Mixed precision (or bf16 if available)
        
        # Logging
        logging_steps=10,
        logging_dir='./logs',
        report_to='wandb',                  # Use Weights & Biases
        
        # Evaluation
        evaluation_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=3,                 # Keep only 3 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        
        # Other
        seed=42,
        dataloader_num_workers=2,
    )
```

---

## Stage 4: Training Pipeline

### 4.1 Custom Data Collator

```python
from transformers import DataCollatorForLanguageModeling

class ReportDataCollator:
    """
    Custom collator for instruction-formatted reports.
    """
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        """
        Tokenize and prepare batch.
        """
        # Extract text
        texts = [ex['text'] for ex in examples]
        
        # Tokenize
        batch = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Labels = input_ids (causal LM)
        batch['labels'] = batch['input_ids'].clone()
        
        return batch
```

### 4.2 Training Script

```python
from transformers import Trainer
import wandb

def train_medgemma(
    train_dataset,
    eval_dataset,
    model,
    tokenizer,
    training_args
):
    """
    Fine-tune MedGemma with LoRA.
    """
    # Initialize W&B
    wandb.init(
        project='medgemma-brain-tumor-reports',
        name='lora-finetuning-v1',
        config=training_args.to_dict()
    )
    
    # Data collator
    data_collator = ReportDataCollator(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("🚀 Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"✓ Training complete! Model saved to {training_args.output_dir}")
    
    return trainer
```

### 4.3 Complete Training Pipeline

```python
def run_full_training_pipeline(
    synthetic_data_file='synthetic_dataset/augmented_pairs.jsonl',
    output_dir='./medgemma_finetuned'
):
    """
    End-to-end training pipeline.
    """
    # Step 1: Load base model
    print("Step 1: Loading MedGemma-4B...")
    model_name = "google/medgemma-4b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map='auto'
    )
    
    # Step 2: Prepare data
    print("\nStep 2: Formatting dataset...")
    formatter = MedGemmaDatasetFormatter(tokenizer)
    formatted_file = 'formatted_training_data.jsonl'
    formatter.process_dataset(synthetic_data_file, formatted_file)
    
    train_data, eval_data = prepare_training_data(formatted_file)
    
    # Step 3: Setup LoRA
    print("\nStep 3: Configuring LoRA...")
    model = setup_lora_model(base_model, tokenizer)
    
    # Step 4: Train
    print("\nStep 4: Training...")
    training_args = get_training_args(output_dir)
    trainer = train_medgemma(
        train_data,
        eval_data,
        model,
        tokenizer,
        training_args
    )
    
    print("\n✅ Training pipeline complete!")
    return model, tokenizer, trainer

# Execute
if __name__ == '__main__':
    model, tokenizer, trainer = run_full_training_pipeline()
```

---

## Stage 5: Inference and Evaluation

### 5.1 Generate Reports from New JSON

```python
class MedGemmaReportGenerator:
    """
    Production report generator using fine-tuned MedGemma.
    """
    
    def __init__(self, model_path, device='cuda'):
        from peft import PeftModel
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "google/medgemma-4b",
            load_in_8bit=True,
            device_map='auto'
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.formatter = MedGemmaDatasetFormatter(self.tokenizer)
        self.model.eval()
    
    def generate_report(self, descriptor, max_new_tokens=512, temperature=0.7):
        """
        Generate report from JSON descriptor.
        
        Args:
            descriptor: Dictionary conforming to schema
            max_new_tokens: Maximum report length
            temperature: Sampling temperature (0.0 = deterministic)
        
        Returns:
            str: Generated report
        """
        # Format input
        user_prompt = self.formatter.format_descriptor_as_prompt(descriptor)
        
        system_prompt = self.formatter.system_prompt
        
        full_prompt = f"""<start_of_turn>system
{system_prompt}<end_of_turn>
<start_of_turn>user
Generate a radiology report based on the following brain tumor analysis:

{user_prompt}
<end_of_turn>
<start_of_turn>model
"""
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1536
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract report (after "model\n")
        report = generated_text.split('<start_of_turn>model')[-1].strip()
        
        return report
```

### 5.2 Evaluation Metrics

```python
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score as bert_score

class ReportEvaluator:
    """
    Evaluate generated reports against ground truth.
    """
    
    def __init__(self):
        self.rouge = Rouge()
    
    def evaluate(self, generated_report, reference_report):
        """
        Compute multiple metrics.
        
        Returns:
            Dictionary of scores
        """
        # Tokenize
        gen_tokens = generated_report.split()
        ref_tokens = reference_report.split()
        
        # BLEU
        bleu = sentence_bleu([ref_tokens], gen_tokens)
        
        # ROUGE
        try:
            rouge_scores = self.rouge.get_scores(generated_report, reference_report)[0]
        except:
            rouge_scores = {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}
        
        # BERTScore (semantic similarity)
        P, R, F1 = bert_score(
            [generated_report],
            [reference_report],
            lang='en',
            model_type='microsoft/deberta-base-mnli'
        )
        
        return {
            'bleu': float(bleu),
            'rouge1_f': rouge_scores['rouge-1']['f'],
            'rouge2_f': rouge_scores['rouge-2']['f'],
            'rougeL_f': rouge_scores['rouge-l']['f'],
            'bertscore_f1': float(F1[0])
        }
    
    def factual_consistency_check(self, descriptor, report):
        """
        Verify report doesn't hallucinate facts.
        
        Returns:
            (consistent: bool, errors: list)
        """
        errors = []
        
        # Volume check (with tolerance)
        import re
        stated_vol = descriptor['segmentation_results']['volumetric_analysis']['total_tumor_volume_mm3']
        vol_matches = re.findall(r'(\d+\.?\d*)\s*mm³', report)
        
        if vol_matches:
            for vol_str in vol_matches:
                vol = float(vol_str)
                if abs(vol - stated_vol) / stated_vol > 0.05:  # 5% tolerance
                    errors.append(f"Volume mismatch: {vol} vs {stated_vol}")
        
        # Region mentions
        regions = [r['region_name'] for r in descriptor['anatomical_mapping']['affected_regions'][:3]]
        for region in regions:
            if region.lower() not in report.lower():
                errors.append(f"Missing critical region: {region}")
        
        # Hemisphere
        hemisphere = descriptor['anatomical_mapping']['hemisphere']
        if hemisphere != 'bilateral' and hemisphere.lower() not in report.lower():
            errors.append(f"Missing hemisphere: {hemisphere}")
        
        return len(errors) == 0, errors
```

### 5.3 Batch Evaluation

```python
def evaluate_model_on_testset(
    generator,
    test_jsonl,
    output_file='evaluation_results.json'
):
    """
    Evaluate fine-tuned model on hold-out test set.
    """
    evaluator = ReportEvaluator()
    results = []
    
    with open(test_jsonl, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            descriptor = data['descriptor']
            reference = data['report']
            
            # Generate
            generated = generator.generate_report(descriptor, temperature=0.1)
            
            # Evaluate
            scores = evaluator.evaluate(generated, reference)
            consistent, errors = evaluator.factual_consistency_check(descriptor, generated)
            
            result = {
                'case_id': descriptor['patient_info']['case_id'],
                'scores': scores,
                'factually_consistent': consistent,
                'errors': errors,
                'generated_report': generated,
                'reference_report': reference
            }
            
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i+1} samples...")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Aggregate metrics
    avg_bleu = sum(r['scores']['bleu'] for r in results) / len(results)
    avg_bertscore = sum(r['scores']['bertscore_f1'] for r in results) / len(results)
    consistency_rate = sum(r['factually_consistent'] for r in results) / len(results)
    
    print(f"\n📊 Evaluation Results:")
    print(f"Average BLEU: {avg_bleu:.3f}")
    print(f"Average BERTScore F1: {avg_bertscore:.3f}")
    print(f"Factual Consistency Rate: {consistency_rate:.1%}")
    
    return results
```

---

## Stage 6: Safety and Bias Mitigation

### 6.1 Medical Safety Constraints

```python
class MedicalSafetyFilter:
    """
    Prevent harmful or misleading medical text.
    """
    
    def __init__(self):
        # Prohibited phrases
        self.prohibited = [
            'diagnostic',
            'diagnosis of',
            'confirmed',
            'treatment recommendation',
            'prognosis is'
        ]
        
        # Required disclaimers
        self.imaging_only_phrases = [
            'imaging findings',
            'consistent with',
            'suggestive of',
            'appearance compatible with'
        ]
    
    def validate_report(self, report):
        """
        Check for safety violations.
        
        Returns:
            (safe: bool, warnings: list)
        """
        warnings = []
        
        report_lower = report.lower()
        
        # Check prohibited terms
        for phrase in self.prohibited:
            if phrase in report_lower:
                warnings.append(f"Uses prohibited phrase: '{phrase}'")
        
        # Check for appropriate hedging
        has_hedging = any(phrase in report_lower for phrase in self.imaging_only_phrases)
        if not has_hedging:
            warnings.append("Lacks appropriate clinical hedging")
        
        # Check for unsupported claims
        if 'definitive' in report_lower or 'certain' in report_lower:
            warnings.append("Makes overly definitive claims")
        
        return len(warnings) == 0, warnings
```

---

## Production Deployment

### API Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model once at startup
generator = MedGemmaReportGenerator('./medgemma_finetuned')
safety_filter = MedicalSafetyFilter()

@app.route('/generate_report', methods=['POST'])
def generate_report_api():
    """
    API endpoint for report generation.
    
    POST body: JSON descriptor
    Returns: Generated report + metadata
    """
    try:
        descriptor = request.get_json()
        
        # Generate
        report = generator.generate_report(descriptor, temperature=0.1)
        
        # Safety check
        is_safe, warnings = safety_filter.validate_report(report)
        
        return jsonify({
            'success': True,
            'report': report,
            'safety': {
                'passed': is_safe,
                'warnings': warnings
            },
            'model_version': 'medgemma-4b-lora-v1.0'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## Continual Learning from Feedback

```python
class FeedbackCollector:
    """
    Collect clinician feedback for continual improvement.
    """
    
    def save_feedback(self, case_id, descriptor, generated_report, corrections):
        """
        Store feedback for future retraining.
        
        Args:
            corrections: Dict with corrected report and annotations
        """
        feedback_entry = {
            'case_id': case_id,
            'descriptor': descriptor,
            'generated_report': generated_report,
            'corrected_report': corrections['report'],
            'annotations': corrections.get('annotations', []),
            'timestamp': datetime.utcnow().isoformat(),
            'clinician_id': corrections.get('clinician_id')
        }
        
        # Append to feedback database
        with open('feedback_database.jsonl', 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
    
    def retrain_with_feedback(self, min_samples=100):
        """
        Periodically retrain with accumulated feedback.
        """
        # Load feedback
        feedback_data = []
        with open('feedback_database.jsonl', 'r') as f:
            for line in f:
                feedback_data.append(json.loads(line))
        
        if len(feedback_data) < min_samples:
            print(f"Not enough feedback ({len(feedback_data)}/{min_samples})")
            return
        
        # Format as training data
        formatter = MedGemmaDatasetFormatter(tokenizer)
        training_examples = []
        
        for feedback in feedback_data:
            example = formatter.create_training_example(
                feedback['descriptor'],
                feedback['corrected_report']  # Use corrected version
            )
            training_examples.append(example)
        
        # Save and retrain
        with open('feedback_training_data.jsonl', 'w') as f:
            for ex in training_examples:
                f.write(json.dumps(ex) + '\n')
        
        print(f"✓ Prepared {len(training_examples)} feedback examples for retraining")
```

---

## Summary

### Key Decisions

1. **Model**: MedGemma-4B (medical domain pre-trained)
2. **Method**: LoRA (PEFT) - efficient for 8GB GPU
3. **Data**: 10,000+ synthetic JSON→Report pairs
4. **Training**: 3 epochs, batch size 1, gradient accumulation
5. **Safety**: Medical terminology validation, factual consistency checks

### Expected Performance

| Metric | Target |
|--------|--------|
| BLEU-4 | > 0.35 |
| BERTScore F1 | > 0.75 |
| Factual Consistency | > 95% |
| Inference Time | < 5s per report |

### Next Steps

1. Run full training pipeline
2. Evaluate on hold-out test set
3. Deploy as API service
4. Collect clinician feedback
5. Iteratively improve with continual learning

---

## Complete Example

See `src/llm/train_medgemma.py` for full implementation.

```bash
# Train
python src/llm/train_medgemma.py \
    --data_file synthetic_dataset/augmented_pairs.jsonl \
    --output_dir ./medgemma_finetuned \
    --epochs 3

# Evaluate
python src/llm/evaluate_medgemma.py \
    --model_path ./medgemma_finetuned \
    --test_file test_cases.jsonl

# Deploy
python src/llm/serve_model.py \
    --model_path ./medgemma_finetuned \
    --port 5000
```

---

**End of LLM Fine-Tuning Guide**
