#!/bin/bash
# Score the generated text and report TPR at 0% / 5% FPR and AUROC:
# 5 watermarking algorithms x 4 FactEHR tasks, with Llama-3-8B-Instruct.
#
# Run run_generation.sh first. Detection scores are cached next to the generation
# output in ../logs/<ALGORITHM>/, so a second run can add --metrics_only to skip the GPU.

set -e

MODEL=llama3
DATASETS="ehr_procedures ehr_progress_note ehr_nursing_note ehr_discharge_summary"

for DATASET in $DATASETS; do
    echo "=============================="
    echo "Dataset: $DATASET"
    echo "=============================="

    python detection_evaluation.py --algorithm KGW      --model $MODEL --dataset $DATASET --gamma 0.5 --delta 2
    python detection_evaluation.py --algorithm SWEET    --model $MODEL --dataset $DATASET --gamma 0.5 --delta 2 --entropy 0.9
    python detection_evaluation.py --algorithm DIP      --model $MODEL --dataset $DATASET --alpha 0.45
    python detection_evaluation.py --algorithm Unbiased --model $MODEL --dataset $DATASET --type gamma
    python detection_evaluation.py --algorithm SIR      --model $MODEL --dataset $DATASET
done
