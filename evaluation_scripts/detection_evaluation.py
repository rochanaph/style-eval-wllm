#!/usr/bin/env python3
"""
Detection Evaluation Script

This script calculates detection scores for the generated texts and reports TPR at a
target FPR together with AUROC.

Usage:
    python detection_evaluation.py --algorithm KGW --model llama3 --dataset ehr_progress_note --gamma 0.5 --delta 2
    python detection_evaluation.py --algorithm SWEET --model llama3 --dataset ehr_procedures --gamma 0.5 --delta 2 --entropy 0.9
    python detection_evaluation.py --algorithm DIP --model llama3 --dataset ehr_procedures --alpha 0.45 --metrics_only
    python detection_evaluation.py --algorithm Unbiased --model llama3 --dataset ehr_procedures --type gamma
"""

import os
import gc
import sys
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from constants import PROJECT_ROOT
from our_utils import ConfigManager, FileManager, ModelLoader
from watermark.auto_watermark import AutoWatermark
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator


def detect_watermark_single_process(dump_path, algorithm, model_type, dataset, params=None):
    """
    Calculate detection scores for the generated texts.
    """
    print(f"Try to load: {algorithm} {model_type} {dataset}")
    print(f"Parameters: {params}")
    file_name = FileManager.build_filename(algorithm, params, model_type, dataset)
    print(f"Loaded watermarked texts from file: {file_name}")
    watermark_file_path = os.path.join(dump_path, file_name)

    with open(watermark_file_path, 'rb') as f:
        watermarked_texts, unwatermarked_texts, natural_texts = pickle.load(f)

    # Truncate natural text to match watermarked text length (character-level)
    # This is necessary because natural texts are often much longer
    natural_texts = [n[:len(w)] for n, w in zip(natural_texts, watermarked_texts)]

    config = ConfigManager.load_config(algorithm, params.copy() if params else {})
    ConfigManager.save_config(algorithm, config)
    config_path = os.path.join(PROJECT_ROOT, "config", f"{algorithm}.json")

    # Initialize watermark detector
    transformers_config = ModelLoader.load_model(model_type, max_new_tokens=200, min_new_tokens=20)

    my_watermark = AutoWatermark.load(
        algorithm,
        algorithm_config=config_path,
        transformers_config=transformers_config
    )

    print(f"Processing {len(watermarked_texts)} watermarked texts...")

    debug_print = True

    # Detect watermarked texts
    w_scores = []
    for text in tqdm(watermarked_texts, desc="Detecting watermarked texts"):
        if debug_print:
            print("w:", text)
            debug_print = False
        w_scores.append(my_watermark.detect_watermark(text, return_dict=True)['score'])
        torch.cuda.empty_cache()
        gc.collect()

    print(f'Watermarked Score [{min(w_scores) if w_scores else "N/A"}, {max(w_scores) if w_scores else "N/A"}]')

    # Save w_scores backup
    w_scores_backup_path = os.path.join(dump_path, f'{file_name}-W_SCORES.pkl')
    with open(w_scores_backup_path, 'wb') as f:
        pickle.dump(w_scores, f)
    print(f"Saved w_scores backup to {w_scores_backup_path}")

    print(f"Processing {len(unwatermarked_texts)} unwatermarked texts...")
    uw_scores = []
    debug_print = True
    for text in tqdm(unwatermarked_texts, desc="Detecting unwatermarked texts"):
        if debug_print:
            print("uw:", text)
            debug_print = False
        uw_scores.append(my_watermark.detect_watermark(text, return_dict=True)['score'])
        torch.cuda.empty_cache()
        gc.collect()

    # Save uw_scores backup
    uw_scores_backup_path = os.path.join(dump_path, f'{file_name}-UW_SCORES.pkl')
    with open(uw_scores_backup_path, 'wb') as f:
        pickle.dump(uw_scores, f)
    print(f"Saved uw_scores backup to {uw_scores_backup_path}")

    print(f"Processing {len(natural_texts)} natural texts...")
    # Detect natural texts
    n_scores = []
    debug_print = True
    for text in tqdm(natural_texts, desc="Detecting natural texts"):
        if debug_print:
            print("n:", text)
            debug_print = False
        n_scores.append(my_watermark.detect_watermark(text, return_dict=True)['score'])
        torch.cuda.empty_cache()
        gc.collect()

    print(f'Natural Score [{min(n_scores) if n_scores else "N/A"}, {max(n_scores) if n_scores else "N/A"}]')

    # Save n_scores backup
    n_scores_backup_path = os.path.join(dump_path, f'{file_name}-N_SCORES.pkl')
    with open(n_scores_backup_path, 'wb') as f:
        pickle.dump(n_scores, f)
    print(f"Saved n_scores backup to {n_scores_backup_path}")

    # Filter out NaN scores (synchronized filtering - skip index i in both if either has NaN)
    w_scores_original_count = len(w_scores)
    n_scores_original_count = len(n_scores)

    # Build valid indices where BOTH scores are not NaN
    valid_indices = []
    w_nan_indices = []
    n_nan_indices = []

    for i in range(len(w_scores)):
        w_is_nan = isinstance(w_scores[i], float) and np.isnan(w_scores[i])
        n_is_nan = isinstance(n_scores[i], float) and np.isnan(n_scores[i])

        if w_is_nan:
            w_nan_indices.append(i)
        if n_is_nan:
            n_nan_indices.append(i)

        # Only include index if BOTH are valid (not NaN)
        if not w_is_nan and not n_is_nan:
            valid_indices.append(i)

    # Filter both arrays using the same valid indices
    w_scores_clean = [w_scores[i] for i in valid_indices]
    n_scores_clean = [n_scores[i] for i in valid_indices]

    total_skipped = w_scores_original_count - len(valid_indices)
    w_nan_count = len(w_nan_indices)
    n_nan_count = len(n_nan_indices)

    print(f"\n{'='*60}")
    print("NaN Score Filtering (Synchronized)")
    print(f"{'='*60}")
    print(f"Total pairs: {w_scores_original_count}")
    print(f"Valid pairs (both non-NaN): {len(valid_indices)}")
    print(f"Skipped pairs (either has NaN): {total_skipped} ({total_skipped/w_scores_original_count*100:.2f}%)")
    print(f"\nBreakdown:")
    print(f"  Watermarked NaN count: {w_nan_count} ({w_nan_count/w_scores_original_count*100:.2f}%)")
    print(f"  Natural NaN count: {n_nan_count} ({n_nan_count/n_scores_original_count*100:.2f}%)")
    if w_nan_indices:
        print(f"  Watermarked NaN indices (first 10): {w_nan_indices[:10]}")
    if n_nan_indices:
        print(f"  Natural NaN indices (first 10): {n_nan_indices[:10]}")
    print(f"{'='*60}\n")

    # Check if we have valid scores after filtering
    if len(w_scores_clean) == 0 or len(n_scores_clean) == 0:
        print("ERROR: No valid score pairs remaining after NaN filtering! Cannot proceed with detection metrics.")
        return None, None, None

    # Update scores to use cleaned versions
    w_scores = w_scores_clean
    n_scores = n_scores_clean

    # Save detection scores
    score_file_path = os.path.join(dump_path, f'{file_name}-SCORES.pkl')
    with open(score_file_path, 'wb') as f:
        pickle.dump((w_scores, uw_scores, n_scores), f)

    print(f"Saved scores to {score_file_path}")

    # Clean up
    del my_watermark
    torch.cuda.empty_cache()
    gc.collect()

    return w_scores, uw_scores, n_scores


def calculate_detection_metrics(w_scores, uw_scores, n_scores, target_fpr=0.0):
    """
    Calculate TPR and AUROC from detection scores.
    """
    calculator = DynamicThresholdSuccessRateCalculator(
        labels=['TPR', 'FPR', 'F1', 'AUROC', 'AUROC_fpr', 'AUROC_tpr', 'Threshold'],
        rule='target_fpr',
        target_fpr=target_fpr
    )

    return calculator.calculate(w_scores, n_scores)


def load_existing_scores(dump_path, algorithm, model_type, dataset, params=None):
    """
    Load existing detection scores from file if available.
    """
    file_name = FileManager.build_filename(algorithm, params, model_type, dataset)
    score_file_path = os.path.join(dump_path, f'{file_name}-SCORES.pkl')

    if not os.path.exists(score_file_path):
        print(f"No existing scores found at: {score_file_path}")
        return None, None, None

    with open(score_file_path, 'rb') as f:
        w_scores, uw_scores, n_scores = pickle.load(f)
    print(f"Loaded existing scores from: {score_file_path}")
    return w_scores, uw_scores, n_scores


def main():
    parser = argparse.ArgumentParser(description='Detection Evaluation: TPR and AUROC calculation')
    parser.add_argument('--algorithm', type=str, required=True,
                       choices=['KGW', 'SWEET', 'DIP', 'Unbiased', 'SIR'],
                       help='Watermarking algorithm')
    parser.add_argument('--model', type=str, required=True,
                       choices=['llama3', 'med42', 'medgemma', 'gemma'],
                       help='Model type')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ehr_procedures', 'ehr_progress_note',
                                'ehr_nursing_note', 'ehr_discharge_summary'],
                       help='Dataset name')
    parser.add_argument('--dump_path', type=str, default=None,
                       help='Path to the generation output directory (default: logs/{algorithm})')

    # Algorithm specific parameters
    parser.add_argument('--gamma', type=float, help='Gamma parameter for KGW/SWEET')
    parser.add_argument('--delta', type=int, help='Delta parameter for KGW/SWEET')
    parser.add_argument('--entropy', type=float, default=0.9, help='Entropy parameter for SWEET')
    parser.add_argument('--alpha', type=float, help='Alpha parameter for DIP')
    parser.add_argument('--type', type=str, help='Type parameter for Unbiased, "gamma" or "delta"')

    # Detection parameters
    parser.add_argument('--target_fpr', type=float, default=0.0, help='Target FPR for threshold selection')

    # Options
    parser.add_argument('--use_existing', action='store_true',
                       help='Use existing detection scores if available')
    parser.add_argument('--output', type=str, help='Output file path for results')
    parser.add_argument('--metrics_only', action='store_true',
                       help='Only calculate metrics from existing scores (no detection)')

    args = parser.parse_args()

    # Build parameters dictionary
    params = {}
    if args.gamma is not None:
        params['gamma'] = args.gamma
    if args.delta is not None:
        params['delta'] = args.delta
    if args.algorithm == 'SWEET' and args.entropy is not None:
        params['entropy'] = args.entropy
        params['entropy_threshold'] = args.entropy
    if args.alpha is not None:
        params['alpha'] = args.alpha
    if args.type is not None:
        params['type'] = args.type

    # Set default dump path
    if args.dump_path is None:
        args.dump_path = os.path.join(PROJECT_ROOT, "logs", args.algorithm)

    print(f"Detection Evaluation for {args.algorithm} {args.model} {args.dataset}")
    print(f"Parameters: {params}")
    print(f"Dump path: {args.dump_path}")
    print(f"Target FPR: {args.target_fpr}")

    w_scores, uw_scores, n_scores = None, None, None

    # Try to load existing scores first
    if args.use_existing or args.metrics_only:
        w_scores, uw_scores, n_scores = load_existing_scores(
            args.dump_path, args.algorithm, args.model, args.dataset, params
        )

    # Calculate detection scores if not available or not using existing
    if not args.metrics_only and (w_scores is None or n_scores is None):
        print("\n=== Calculating Detection Scores ===")
        w_scores, uw_scores, n_scores = detect_watermark_single_process(
            dump_path=args.dump_path,
            algorithm=args.algorithm,
            model_type=args.model,
            dataset=args.dataset,
            params=params
        )

    if w_scores is None or n_scores is None:
        print("Error: Could not load or calculate detection scores")
        return
    if len(w_scores) == 0 or len(n_scores) == 0:
        print("Error: No valid scores available after filtering (all scores are NaN)")
        return
    if len(w_scores) != len(n_scores):
        print(f"Error: Watermarked and natural scores have different lengths after filtering: {len(w_scores)} vs {len(n_scores)}")
        return

    print(f"\n=== Calculating Detection Metrics ===")
    print(f"Loaded {len(w_scores)} watermarked scores and {len(n_scores)} natural scores")

    # Calculate metrics for both 0% and 5% FPR by default
    target_fprs = [0.0, 0.05] if args.target_fpr == 0.0 else [args.target_fpr]

    results_by_fpr = {}
    for target_fpr in target_fprs:
        results_by_fpr[target_fpr] = calculate_detection_metrics(w_scores, uw_scores, n_scores, target_fpr)

    # Use the primary target_fpr for main results
    result = results_by_fpr[args.target_fpr]

    results = {
        'algorithm': args.algorithm,
        'model': args.model,
        'dataset': args.dataset,
        'target_fpr': args.target_fpr,
        'tpr': result['TPR'],
        'fpr': result['FPR'],
        'f1': result['F1'],
        'auroc': result['AUROC'],
        'threshold': result['Threshold'],
        'num_watermarked_scores': len(w_scores),
        'num_natural_scores': len(n_scores),
        'watermarked_score_range': [min(w_scores), max(w_scores)],
        'natural_score_range': [min(n_scores), max(n_scores)],
        **params
    }

    # Add results for all FPR values
    if len(target_fprs) > 1:
        results['metrics_by_fpr'] = {
            f'{int(fpr*100)}%': {
                'tpr': results_by_fpr[fpr]['TPR'],
                'fpr': results_by_fpr[fpr]['FPR'],
                'f1': results_by_fpr[fpr]['F1'],
                'threshold': results_by_fpr[fpr]['Threshold']
            }
            for fpr in target_fprs
        }

    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n=== Detection Results ===")
    print(f"AUROC: {result['AUROC']:.4f}")

    # Print results for all FPR thresholds
    if len(target_fprs) > 1:
        print("\nMetrics at different FPR thresholds:")
        for target_fpr in target_fprs:
            fpr_result = results_by_fpr[target_fpr]
            print(f"  At {target_fpr*100:.0f}% FPR: TPR={fpr_result['TPR']:.4f}, F1={fpr_result['F1']:.4f}, Threshold={fpr_result['Threshold']:.4f}, Actual FPR={fpr_result['FPR']:.4f}")
    else:
        print(f"TPR (at {args.target_fpr*100}% FPR): {result['TPR']:.4f}")
        print(f"F1 Score: {result['F1']:.4f}")
        print(f"Threshold: {result['Threshold']:.4f}")
        print(f"Actual FPR: {result['FPR']:.4f}")

    print(f"\nWatermarked scores: [{min(w_scores):.4f}, {max(w_scores):.4f}]")
    print(f"Natural scores: [{min(n_scores):.4f}, {max(n_scores):.4f}]")

    return results


if __name__ == "__main__":
    main()
