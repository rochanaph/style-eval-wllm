#!/usr/bin/env python3
"""
Style and Quality Evaluation

Evaluates watermarked and unwatermarked text against the natural (reference) text for
EACH individual data point, across the JSON files produced by text_generation.py:
- METEOR
- BERTScore F1
- Gunning Fog Index (GFI)
- Flesch Reading Ease (FRE)
- Earth Mover's Distance on POS tags, against the natural text (EMD)
- HELP, the fitted proxy for the LLM judge Helpfulness score, on the same 1-5 scale

Usage:
    python style_evaluation.py --task ehr_nursing_note
    python style_evaluation.py --task ehr_procedures --algorithms KGW SWEET
"""

import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import PROJECT_ROOT
import evaluate

# Import shared style functions
from style_functions import (
    clean_delimiter_artifacts,
    flesch_reading_ease,
    gunning_fog_index,
    emd_pos,
    load_tokenizer,
    truncate_to_n_tokens,
    parse_filename,
    calculate_help_score
)

ALGORITHMS = ['KGW', 'SWEET', 'DIP', 'Unbiased', 'SIR']
TASKS = ['ehr_procedures', 'ehr_progress_note', 'ehr_nursing_note', 'ehr_discharge_summary']

# Generation output names, matching our_utils.FileManager.build_filename with the
# parameters used by run_generation.sh.
FILE_PATTERNS = {
    'KGW': 'KGW-g0.5-d2-{model}-{task}.pkl.json',
    'SWEET': 'SWEET-e0.9-g0.5-d2-{model}-{task}.pkl.json',
    'DIP': 'DIP-a0.45-{model}-{task}.pkl.json',
    'Unbiased': 'Unbiased-tgamma-{model}-{task}.pkl.json',
    'SIR': 'SIR-{model}-{task}.pkl.json',
}


# ==============================================================================
# METRICS CALCULATION
# ==============================================================================

def calculate_metrics_for_record(record, tokenizer, bertscore_metric, meteor_metric):
    """
    Calculate all metrics for a single record.

    Returns:
        Dictionary with all metric values
    """
    natural_text = record.get('natural_text', '')
    watermarked_text = record.get('watermarked_text', '')
    unwatermarked_text = record.get('unwatermarked_text', '')

    # Truncate natural text to 200 tokens, to match max_new_tokens in generation
    natural_text = truncate_to_n_tokens(natural_text, tokenizer, max_tokens=200)

    metrics = {}

    # Clean delimiter artifacts (//) for the style metrics only
    natural_text_clean = clean_delimiter_artifacts(natural_text)
    watermarked_text_clean = clean_delimiter_artifacts(watermarked_text)
    unwatermarked_text_clean = clean_delimiter_artifacts(unwatermarked_text)

    # Readability: Gunning Fog Index
    metrics['GFI_U'] = gunning_fog_index(unwatermarked_text_clean)
    metrics['GFI_W'] = gunning_fog_index(watermarked_text_clean)

    # Structural similarity: EMD on POS tags, against the natural text
    metrics['EMD_U_N'] = emd_pos(unwatermarked_text_clean, natural_text_clean)
    metrics['EMD_W_N'] = emd_pos(watermarked_text_clean, natural_text_clean)

    # Accuracy: METEOR
    meteor_u = meteor_metric.compute(
        predictions=[unwatermarked_text],
        references=[[natural_text]]
    )
    metrics['METEOR_U'] = meteor_u['meteor']

    meteor_w = meteor_metric.compute(
        predictions=[watermarked_text],
        references=[[natural_text]]
    )
    metrics['METEOR_W'] = meteor_w['meteor']

    # Accuracy: BERTScore F1 (reported, not used by HELP)
    bertscore_u = bertscore_metric.compute(
        predictions=[unwatermarked_text],
        references=[natural_text],
        lang="en"
    )
    metrics['BERTScore_F1_U'] = bertscore_u['f1'][0]

    bertscore_w = bertscore_metric.compute(
        predictions=[watermarked_text],
        references=[natural_text],
        lang="en"
    )
    metrics['BERTScore_F1_W'] = bertscore_w['f1'][0]

    # Readability: Flesch Reading Ease (reported, not used by HELP)
    metrics['FRE_U'] = flesch_reading_ease(unwatermarked_text_clean)
    metrics['FRE_W'] = flesch_reading_ease(watermarked_text_clean)

    # Helpfulness. The regression outputs [0,1], map to the 1-5 scale the LLM judge uses.
    help_u_raw = calculate_help_score(
        metrics['METEOR_U'],
        metrics['EMD_U_N'],
        metrics['GFI_U']
    )
    metrics['HELP_U'] = help_u_raw * 4 + 1

    help_w_raw = calculate_help_score(
        metrics['METEOR_W'],
        metrics['EMD_W_N'],
        metrics['GFI_W']
    )
    metrics['HELP_W'] = help_w_raw * 4 + 1

    # Drop in helpfulness caused by the watermark
    metrics['Delta_HELP'] = metrics['HELP_U'] - metrics['HELP_W']

    return metrics


def process_json_file(json_path, bertscore_metric, meteor_metric):
    """
    Process a single JSON file and return metrics for each individual record.

    Returns:
        DataFrame with one row per record
    """
    scheme, task, model = parse_filename(json_path)

    print(f"\n{'='*80}")
    print(f"Processing: {scheme} / {task} / {model}")
    print(f"{'='*80}\n")

    tokenizer = load_tokenizer(model)

    print(f"Loading data from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records\n")

    results = []

    for idx, record in enumerate(tqdm(data, desc="Processing records")):
        metrics = calculate_metrics_for_record(record, tokenizer, bertscore_metric, meteor_metric)

        row = {
            'sample_id': idx,
            'scheme': scheme,
            'task': task,
            'model': model,
        }
        row.update(metrics)

        results.append(row)

    df = pd.DataFrame(results)

    # Reorder columns for better readability
    meta_cols = ['sample_id', 'scheme', 'task', 'model']
    help_cols = ['HELP_U', 'HELP_W', 'Delta_HELP']
    accuracy_cols = ['METEOR_U', 'METEOR_W', 'BERTScore_F1_U', 'BERTScore_F1_W']
    syntactic_cols = ['EMD_U_N', 'EMD_W_N']
    readability_cols = ['GFI_U', 'GFI_W', 'FRE_U', 'FRE_W']

    return df[meta_cols + help_cols + accuracy_cols + syntactic_cols + readability_cols]


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Style and quality evaluation of generated text')
    parser.add_argument('--task', required=True, choices=TASKS, help='FactEHR task to evaluate')
    parser.add_argument('--algorithms', nargs='+', default=ALGORITHMS, choices=ALGORITHMS,
                        help='Watermarking algorithms to include (default: all)')
    parser.add_argument('--model', default='llama3',
                        choices=['llama3', 'med42', 'medgemma', 'gemma'],
                        help='Model whose generations to evaluate')
    parser.add_argument('--logdir', default=os.path.join(PROJECT_ROOT, 'logs'),
                        help='Directory holding the per-algorithm generation output')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: results/style_<task>.csv)')

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(PROJECT_ROOT, 'results', f'style_{args.task}.csv')

    json_files = [
        os.path.join(args.logdir, algorithm,
                     FILE_PATTERNS[algorithm].format(model=args.model, task=args.task))
        for algorithm in args.algorithms
    ]

    print(f"\n{'='*80}")
    print(f"Processing {len(json_files)} JSON files")
    print(f"{'='*80}\n")

    # Load evaluation metrics once, they are expensive to build
    print("Loading evaluation metrics...")
    bertscore_metric = evaluate.load('bertscore')
    meteor_metric = evaluate.load('meteor')

    all_dfs = []
    for json_path in json_files:
        if not os.path.exists(json_path):
            print(f"WARNING: File not found: {json_path}")
            print("Skipping...\n")
            continue

        all_dfs.append(process_json_file(json_path, bertscore_metric, meteor_metric))

    if not all_dfs:
        print("ERROR: No files were successfully processed!")
        return

    print(f"\n{'='*80}")
    print(f"Concatenating results from {len(all_dfs)} files")
    print(f"{'='*80}\n")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Add continuous uid column
    combined_df.insert(0, 'uid', range(len(combined_df)))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined_df.to_csv(args.output, index=False)

    print(f"\n{'='*80}")
    print(f"Results Summary")
    print(f"{'='*80}")
    print(f"Total records across all files: {len(combined_df)}")
    print(f"Number of files processed: {len(all_dfs)}")
    print(f"\nSaved combined results to: {args.output}")
    print(f"  Shape: {combined_df.shape}")

    print("\nMean per scheme:")
    summary = combined_df.groupby(['scheme', 'task'])[
        ['HELP_U', 'HELP_W', 'Delta_HELP', 'METEOR_U', 'METEOR_W', 'EMD_U_N', 'EMD_W_N', 'GFI_U', 'GFI_W']
    ].mean()
    print(summary.to_string())
    print()


if __name__ == '__main__':
    main()
