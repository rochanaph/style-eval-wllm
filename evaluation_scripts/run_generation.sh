#!/bin/bash
# Generate watermarked and unwatermarked text for the main experiment:
# 5 watermarking algorithms x 4 FactEHR tasks, with Llama-3-8B-Instruct.
#
# Outputs go to ../logs/<ALGORITHM>/.
# SIR needs the models from `python ../download_sir_models.py` first.

set -e

MODEL=llama3
DATASETS="ehr_procedures ehr_progress_note ehr_nursing_note ehr_discharge_summary"

for DATASET in $DATASETS; do
    echo "=============================="
    echo "Dataset: $DATASET"
    echo "=============================="

    python text_generation.py --algorithm KGW      --model $MODEL --dataset $DATASET --gamma 0.5 --delta 2
    python text_generation.py --algorithm SWEET    --model $MODEL --dataset $DATASET --gamma 0.5 --delta 2 --entropy 0.9
    python text_generation.py --algorithm DIP      --model $MODEL --dataset $DATASET --alpha 0.45
    python text_generation.py --algorithm Unbiased --model $MODEL --dataset $DATASET --type gamma
    python text_generation.py --algorithm SIR      --model $MODEL --dataset $DATASET
done
