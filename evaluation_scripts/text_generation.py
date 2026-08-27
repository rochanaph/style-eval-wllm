#!/usr/bin/env python3
"""
Text Generation Script

This script generates watermarked and unwatermarked text using various watermarking algorithms.

Usage:
    python text_generation.py --algorithm KGW --model llama3 --dataset ehr_procedures --gamma 0.5 --delta 2
    python text_generation.py --algorithm SWEET --model llama3 --dataset ehr_procedures --gamma 0.25 --delta 8 --entropy 0.9
    python text_generation.py --algorithm DIP --model llama3 --dataset ehr_nursing_note --alpha 0.45
    python text_generation.py --algorithm Unbiased --model llama3 --dataset ehr_nursing_note --type gamma
    python text_generation.py --algorithm SIR --model llama3 --dataset ehr_nursing_note
"""

import os
import gc
import json
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import set_seed

# Add parent directory to path to import utilities
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from constants import PROJECT_ROOT
from watermark.auto_watermark import AutoWatermark
from evaluation.dataset import FactEHRDataset
from our_utils import ConfigManager, FileManager, ModelLoader, PromptFormatter

DEFAULT_DEVICE = 'auto'

# Models with a built-in chat template. Their prompts need instruction formatting.
INSTRUCT_MODELS = ['llama3', 'med42', 'medgemma', 'gemma']


# Helper function to process prompts for instruct models
def get_formatted_prompt(raw_prompt, tokenizer):
    """
    Format prompt using chat template if tokenizer is provided.

    Args:
        raw_prompt: The raw instruction prompt
        tokenizer: Model tokenizer (None for base models, provided for instruct models)

    Returns:
        Formatted prompt (with chat template for instruct models, raw for base models)
    """
    # Base models don't have chat templates. Just return the raw prompt
    if tokenizer is None:
        return raw_prompt

    # Instruct models have built-in chat templates
    # Use PromptFormatter to apply the model's native chat template
    return PromptFormatter.format_instruction_prompt(raw_prompt, tokenizer)


# Helper function to extract only the response (remove formatted prompt and preamble)
def extract_response(full_text, formatted_prompt):
    """
    Extract the model's response by removing the formatted prompt.

    Handles different chat template formats:
    - Llama3: <|begin_of_text|>...<|start_header_id|>assistant<|end_header_id|>\n\nRESPONSE
    - Mistral: <s>[INST] prompt [/INST]RESPONSE
    - Gemma/MedGemma: <bos><start_of_turn>user\nprompt<end_of_turn>\n<start_of_turn>model\nRESPONSE
    - Base models: prompt directly followed by response

    Args:
        full_text: The complete generated text (prompt + response)
        formatted_prompt: The formatted prompt to remove

    Returns:
        The extracted response text (with preambles removed)
    """
    # Step 1: Try different extraction methods based on markers
    response = full_text

    # Method 1: Look for "\nmodel\n" marker (Gemma/MedGemma format)
    # This marker appears as plain text when special tokens are stripped
    # Check this FIRST because it's most specific
    if "\nmodel\n" in full_text:
        response = full_text.split("\nmodel\n", 1)[1]

    # Method 2: Look for [/INST] marker (Mistral format)
    # This marker often survives skip_special_tokens=True decoding
    elif "[/INST]" in full_text:
        response = full_text.split("[/INST]", 1)[1]

    # Method 3: Look for "assistant\n\n" marker (Llama3 format)
    elif "assistant\n\n" in full_text:
        response = full_text.split("assistant\n\n", 1)[1]

    # Method 4: Try to remove the formatted prompt directly (for base models)
    elif full_text.startswith(formatted_prompt):
        response = full_text[len(formatted_prompt):]

    # Step 2: Remove preamble by finding ":\n" pattern
    # LLMs typically end preamble with colon followed by newline(s)
    # e.g., "Here are the facts:\n\nThe patient..." or "Atomic facts:\nDrew..."
    if ":\n" in response:
        # Split at first occurrence of ":\n" and take everything after
        parts = response.split(":\n", 1)
        if len(parts) > 1:
            response = parts[1]
            # Strip any leading newlines
            response = response.lstrip('\n')

    return response.strip()


def generate_watermarked_text(algorithm, model_type, dataset, params=None, device=DEFAULT_DEVICE,
                             samples=200000):
    """
    Generate and save watermarked and unwatermarked text.

    Args:
        algorithm: Watermarking algorithm (KGW, SWEET, DIP, Unbiased, SIR)
        model_type: Model type (llama3, med42, medgemma, gemma)
        dataset: Dataset name (ehr_procedures, ehr_progress_note, ehr_nursing_note, ehr_discharge_summary)
        params: Algorithm parameters dictionary
        device: Device to use for generation
        samples: Number of samples to generate
    """
    print(f"Starting generation for {algorithm} with {model_type} on {dataset}")
    set_seed(113)

    # Load dataset
    dataset_path = FileManager.get_dataset_path(dataset)
    my_dataset = FactEHRDataset(dataset_path)

    # Setup model parameters
    max_new_tokens = 200
    min_new_tokens = 20

    # Load model configuration
    transformers_config = ModelLoader.load_model(
        model_type,
        device,
        max_new_tokens,
        min_new_tokens
    )

    config = ConfigManager.load_config(algorithm, params)
    ConfigManager.save_config(algorithm, config)
    config_path = os.path.join(PROJECT_ROOT, "config", f"{algorithm}.json")

    # Initialize watermark
    my_watermark = AutoWatermark.load(
        algorithm,
        algorithm_config=config_path,
        transformers_config=transformers_config
    )

    # Determine if we need instruction formatting (for instruct models)
    use_instruct_format = model_type in INSTRUCT_MODELS
    tokenizer = transformers_config.tokenizer if use_instruct_format else None

    # Generate unwatermarked text first (IMPORTANT: Don't change the order)
    print("Generating unwatermarked texts...")
    unwatermarked_texts = []
    for i in tqdm(range(min(samples, my_dataset.prompt_nums)), desc="Unwatermarked"):
        raw_prompt = my_dataset.get_prompt(i)
        formatted_prompt = get_formatted_prompt(raw_prompt, tokenizer)

        # Retry if output is too short (less than min_new_tokens)
        retry_count = 0
        while retry_count < 3:
            full_output = my_watermark.generate_unwatermarked_text(formatted_prompt)
            response_only = extract_response(full_output, formatted_prompt)

            # Count tokens in response
            response_tokens = transformers_config.tokenizer(response_only, return_tensors="pt", add_special_tokens=False)["input_ids"]
            num_tokens = response_tokens.shape[1] if len(response_tokens.shape) > 1 else 0

            if num_tokens >= min_new_tokens:
                break

            retry_count += 1

        unwatermarked_texts.append(response_only)

    # Generate watermarked text
    print("Generating watermarked texts...")
    watermarked_texts = []
    for i in tqdm(range(min(samples, my_dataset.prompt_nums)), desc="Watermarked"):
        raw_prompt = my_dataset.get_prompt(i)
        formatted_prompt = get_formatted_prompt(raw_prompt, tokenizer)

        # Retry if output is too short (less than min_new_tokens)
        retry_count = 0
        while retry_count < 3:
            full_output = my_watermark.generate_watermarked_text(formatted_prompt)
            response_only = extract_response(full_output, formatted_prompt)

            # Count tokens in response
            response_tokens = transformers_config.tokenizer(response_only, return_tensors="pt", add_special_tokens=False)["input_ids"]
            num_tokens = response_tokens.shape[1] if len(response_tokens.shape) > 1 else 0

            if num_tokens >= min_new_tokens:
                break

            retry_count += 1

        watermarked_texts.append(response_only)

    # Get natural texts
    print("Loading natural texts...")
    natural_texts = []
    for i in tqdm(range(min(samples, my_dataset.natural_text_nums)), desc="Natural"):
        natural_texts.append(my_dataset.get_natural_text(i))

    # Print samples
    print("\n--- SAMPLE OUTPUTS ---")
    print("UNWATERMARKED TEXT:", unwatermarked_texts[0][:200] + "..." if len(unwatermarked_texts[0]) > 200 else unwatermarked_texts[0])
    print("\nWATERMARKED TEXT:", watermarked_texts[0][:200] + "..." if len(watermarked_texts[0]) > 200 else watermarked_texts[0])
    print("\nNATURAL TEXT:", natural_texts[0][:200] + "..." if len(natural_texts[0]) > 200 else natural_texts[0])

    # Save results
    file_name = FileManager.build_filename(
        algorithm,
        params,
        model_type,
        dataset
    )

    output_dir = os.path.join(PROJECT_ROOT, "logs", algorithm)
    os.makedirs(output_dir, exist_ok=True)

    # Save pickle file
    pickle_path = os.path.join(output_dir, file_name)
    with open(pickle_path, 'wb') as f:
        pickle.dump((watermarked_texts, unwatermarked_texts, natural_texts), f)

    print(f"Saved pickle data to: {pickle_path}")

    # Save JSON file for easy inspection
    json_data = []
    for i in range(min(samples, my_dataset.prompt_nums)):
        data_point = {
            "prompt": my_dataset.prompts[i] if i < len(my_dataset.prompts) else None,
            "watermarked_text": watermarked_texts[i] if i < len(watermarked_texts) else None,
            "unwatermarked_text": unwatermarked_texts[i] if i < len(unwatermarked_texts) else None,
            "natural_text": natural_texts[i] if i < len(natural_texts) else None
        }
        json_data.append(data_point)

    json_path = f'{pickle_path}.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"Saved JSON data to: {json_path}")

    # Print statistics
    print(f"\n--- GENERATION STATISTICS ---")
    print(f"Generated {len(watermarked_texts)} watermarked texts")
    print(f"Generated {len(unwatermarked_texts)} unwatermarked texts")
    print(f"Loaded {len(natural_texts)} natural texts")

    # Clean up
    del my_watermark
    torch.cuda.empty_cache()
    gc.collect()

    return pickle_path, json_path


def main():
    parser = argparse.ArgumentParser(description='Generate watermarked text using various algorithms')

    # Required arguments
    parser.add_argument('--algorithm', required=True, choices=['KGW', 'SWEET', 'DIP', 'Unbiased', 'SIR'],
                       help='Watermarking algorithm')
    parser.add_argument('--model', required=True, choices=['llama3', 'med42', 'medgemma', 'gemma'],
                       help='Model type')
    parser.add_argument('--dataset', required=True, choices=['ehr_procedures', 'ehr_progress_note',
                                                             'ehr_nursing_note', 'ehr_discharge_summary'],
                       help='Dataset name')

    # Algorithm-specific parameters
    parser.add_argument('--gamma', type=float, help='Gamma parameter for KGW/SWEET')
    parser.add_argument('--delta', type=int, help='Delta parameter for KGW/SWEET')
    parser.add_argument('--entropy', type=float, help='Entropy parameter for SWEET')
    parser.add_argument('--alpha', type=float, help='Alpha parameter for DIP')
    parser.add_argument('--type', type=str, help='Type parameter for Unbiased, "gamma" or "delta"')

    # Optional arguments
    parser.add_argument('--samples', type=int, default=200000, help='Number of samples to generate')
    parser.add_argument('--device', default='cuda', help='Device to use for generation')
    parser.add_argument('--output', help='Save summary to JSON file')

    args = parser.parse_args()

    # Build parameter dictionary based on algorithm
    params = {}
    if args.algorithm in ['KGW', 'SWEET']:
        if args.gamma is None or args.delta is None:
            parser.error(f'{args.algorithm} requires --gamma and --delta parameters')
        params['gamma'] = args.gamma
        params['delta'] = args.delta
        if args.algorithm == 'SWEET':
            if args.entropy is None:
                parser.error('SWEET requires --entropy parameter')
            params['entropy'] = args.entropy
            params['entropy_threshold'] = args.entropy
    elif args.algorithm == 'DIP':
        if args.alpha is None:
            parser.error('DIP requires --alpha parameter')
        params['alpha'] = args.alpha
    elif args.algorithm == 'Unbiased':
        if args.type is None:
            parser.error('Unbiased requires --type parameter')
        params['type'] = args.type
    elif args.algorithm == 'SIR':
        # SIR has no configurable parameters (fixed delta=1.0, chunk_length=10, etc.)
        pass

    # Generate text
    pickle_path, json_path = generate_watermarked_text(
        algorithm=args.algorithm,
        model_type=args.model,
        dataset=args.dataset,
        params=params,
        device=args.device,
        samples=args.samples
    )

    # Save summary if requested
    if args.output:
        summary = {
            'algorithm': args.algorithm,
            'model': args.model,
            'dataset': args.dataset,
            'parameters': params,
            'samples_generated': args.samples,
            'pickle_file': pickle_path,
            'json_file': json_path,
            'status': 'completed'
        }

        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"Summary saved to: {args.output}")

    print(f"\nGeneration completed successfully!")
    print(f"Pickle file: {pickle_path}")
    print(f"JSON file: {json_path}")

    return 0


if __name__ == '__main__':
    exit(main())
