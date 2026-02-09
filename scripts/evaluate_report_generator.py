"""
Evaluate trained report generator and generate results for paper.
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report_generation.report_generator import ReportGenerator, TemplateBasedGenerator
from src.report_generation.dataset import load_jsonl
from src.report_generation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate report generator")
    parser.add_argument('--test_data', type=str, default='data/reports/test.jsonl',
                        help='Test data path')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='LoRA checkpoint path (if None, uses template baseline)')
    parser.add_argument('--model_name', type=str, default='google/gemma-2b-it',
                        help='Base model name')
    parser.add_argument('--output_dir', type=str, default='results/report_generation',
                        help='Output directory for results')
    parser.add_argument('--num_examples', type=int, default=3,
                        help='Number of example comparisons to save')
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Report Generator Evaluation")
    print("=" * 60)
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = load_jsonl(args.test_data)
    print(f"Test samples: {len(test_data)}")
    print()
    
    # Extract JSON data and reference reports
    json_data_list = [item['json_data'] for item in test_data]
    reference_reports = [item['report'] for item in test_data]
    
    # Evaluate template baseline
    print("Evaluating template baseline...")
    template_gen = TemplateBasedGenerator()
    template_reports = []
    for json_data in tqdm(json_data_list[:100]):  # Limit to 100 for speed
        report = template_gen.generate(json_data, template_idx=0)
        template_reports.append(report)
    
    template_metrics = compute_metrics(
        reference_reports[:100],
        template_reports
    )
    
    print(f"\nTemplate Baseline Metrics:")
    print(f"  BLEU-4: {template_metrics['bleu_4']:.1f}")
    print(f"  ROUGE-L: {template_metrics['rouge_l']:.3f}")
    print(f"  METEOR: {template_metrics['meteor']:.3f}")
    print()
    
    # Evaluate LLM model if checkpoint provided
    if args.checkpoint:
        print(f"Loading LLM model from {args.checkpoint}...")
        llm_gen = ReportGenerator(
            model_name=args.model_name,
            lora_checkpoint=args.checkpoint
        )
        
        print("Generating reports with LLM...")
        llm_reports = []
        for json_data in tqdm(json_data_list[:100]):
            report = llm_gen.generate(json_data)
            llm_reports.append(report)
        
        llm_metrics = compute_metrics(
            reference_reports[:100],
            llm_reports
        )
        
        print(f"\nLLM Model Metrics:")
        print(f"  BLEU-4: {llm_metrics['bleu_4']:.1f}")
        print(f"  ROUGE-L: {llm_metrics['rouge_l']:.3f}")
        print(f"  METEOR: {llm_metrics['meteor']:.3f}")
        print()
        
        # Calculate improvements
        bleu_improvement = llm_metrics['bleu_4'] - template_metrics['bleu_4']
        rouge_improvement = llm_metrics['rouge_l'] - template_metrics['rouge_l']
        meteor_improvement = llm_metrics['meteor'] - template_metrics['meteor']
        
        print(f"Improvements over baseline:")
        print(f"  BLEU-4: +{bleu_improvement:.1f} ({100*bleu_improvement/template_metrics['bleu_4']:.1f}%)")
        print(f"  ROUGE-L: +{rouge_improvement:.3f} ({100*rouge_improvement/template_metrics['rouge_l']:.1f}%)")
        print(f"  METEOR: +{meteor_improvement:.3f} ({100*meteor_improvement/template_metrics['meteor']:.1f}%)")
        print()
    else:
        llm_metrics = None
        llm_reports = None
    
    # Save metrics
    results = {
        'template_baseline': template_metrics,
        'llm_model': llm_metrics,
        'test_samples': len(test_data[:100])
    }
    
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(template_metrics, llm_metrics)
    with open(output_path / 'metrics_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"✅ Metrics saved to {output_path / 'metrics.json'}")
    print(f"✅ LaTeX table saved to {output_path / 'metrics_table.tex'}")
    
    # Save example comparisons
    if llm_reports:
        examples = []
        for i in range(min(args.num_examples, len(test_data))):
            examples.append({
                'json_data': json_data_list[i],
                'reference': reference_reports[i],
                'template': template_reports[i],
                'llm_generated': llm_reports[i]
            })
        
        with open(output_path / 'examples.json', 'w') as f:
            json.dump(examples, f, indent=2)
        
        print(f"✅ {len(examples)} example comparisons saved to {output_path / 'examples.json'}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


def generate_latex_table(template_metrics, llm_metrics=None):
    """Generate LaTeX table for paper."""
    
    table = r"""\begin{table}[htbp]
\caption{Report Generation Performance}
\begin{center}
\begin{tabular}{l c c c}
\toprule
\textbf{Method} & \textbf{BLEU-4} & \textbf{ROUGE-L} & \textbf{METEOR} \\
\midrule
"""
    
    table += f"Template Baseline & {template_metrics['bleu_4']:.1f} & {template_metrics['rouge_l']:.3f} & {template_metrics['meteor']:.3f} \\\\\n"
    
    if llm_metrics:
        table += f"\\textbf{{LoRA Fine-tuned}} & \\textbf{{{llm_metrics['bleu_4']:.1f}}} & \\textbf{{{llm_metrics['rouge_l']:.3f}}} & \\textbf{{{llm_metrics['meteor']:.3f}}} \\\\\n"
    
    table += r"""\bottomrule
\end{tabular}
\label{tab:report_results}
\end{center}
\end{table}"""
    
    return table


if __name__ == '__main__':
    main()
