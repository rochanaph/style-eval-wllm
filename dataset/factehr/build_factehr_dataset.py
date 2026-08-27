#!/usr/bin/env python3
"""
Build the FactEHR generation input files.

The watermarking scripts read one JSONL file per note type, where each line has:

    {"prompt": "<fact decomposition instruction + clinical note>",
     "natural_text": "<GPT-4o reference decomposition>"}

This script joins two FactEHR artifacts on `doc_id`:

  1. the prompted dataset produced by the FactEHR repository script
     `scripts/build_fact_decomp_prompted_dataset.py`, which supplies the prompt
  2. the fact decomposition results CSV, which supplies the GPT-4o reference

`doc_ids.json` in this directory lists the exact notes the paper used, in order, so
the subset is the same for everyone. It holds identifiers only, no clinical text.

See README.md in this directory for how to obtain both inputs.

Usage:
    python build_factehr_dataset.py \
        --prompted_dataset fact_decomposition_delimiter_20250926.jsonl \
        --decompositions fact_decompositions.csv
"""

import os
import json
import argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DOC_IDS_FILE = os.path.join(HERE, "doc_ids.json")

REFERENCE_MODEL = "gpt-4o"


def load_prompts(prompted_dataset):
    """Map doc_id to prompt for every line of the prompted dataset."""
    prompts = {}
    with open(prompted_dataset, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts[data['metadata']['doc_id']] = data['messages'][0]['content']
    return prompts


def build_note_type(prompts, reference, note_type, doc_ids, output_dir):
    """Write one <note_type>.jsonl file."""
    records = []
    no_prompt = []
    no_reference = []

    for doc_id in doc_ids:
        if doc_id not in prompts:
            no_prompt.append(doc_id)
        elif doc_id not in reference.index:
            no_reference.append(doc_id)
        else:
            records.append({
                'prompt': prompts[doc_id],
                'natural_text': reference.loc[doc_id],
            })

    output_file = os.path.join(output_dir, f"{note_type}.jsonl")
    with open(output_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    print(f"{note_type}: wrote {len(records)}/{len(doc_ids)} records to {output_file}")
    if no_prompt:
        print(f"  {len(no_prompt)} doc_ids missing from the prompted dataset, first: {no_prompt[0]}")
    if no_reference:
        print(f"  {len(no_reference)} doc_ids without a {REFERENCE_MODEL} decomposition, first: {no_reference[0]}")


def main():
    parser = argparse.ArgumentParser(description='Build the FactEHR generation input files')
    parser.add_argument('--prompted_dataset', required=True,
                        help='JSONL produced by build_fact_decomp_prompted_dataset.py')
    parser.add_argument('--decompositions', required=True,
                        help='CSV with doc_id, model and decomposition_result columns')
    parser.add_argument('--output_dir', default=HERE,
                        help='Where to write the JSONL files (default: this directory)')
    args = parser.parse_args()

    with open(DOC_IDS_FILE) as f:
        doc_ids_per_note_type = json.load(f)

    prompts = load_prompts(args.prompted_dataset)
    print(f"Loaded {len(prompts)} prompts")

    decompositions = pd.read_csv(args.decompositions)
    reference = decompositions[decompositions['model'] == REFERENCE_MODEL]
    reference = reference.drop_duplicates('doc_id').set_index('doc_id')['decomposition_result']
    print(f"Loaded {len(reference)} {REFERENCE_MODEL} decompositions")

    for note_type, doc_ids in doc_ids_per_note_type.items():
        build_note_type(prompts, reference, note_type, doc_ids, args.output_dir)


if __name__ == '__main__':
    main()
