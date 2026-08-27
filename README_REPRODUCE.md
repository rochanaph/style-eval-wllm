# Reproducing the paper

This repository is a fork of [MarkLLM](https://github.com/THU-BPM/MarkLLM) with the
scripts used in the paper. This document covers **text generation** only, which is the
first stage of the pipeline.

## 1. Install

```bash
conda create -n markllm python=3.11
conda activate markllm
python -m pip install -r requirements.txt
```

The models are gated on the Hugging Face Hub, so log in first:

```bash
huggingface-cli login
```

## 2. Dataset

The clinical notes are **not in this repository**. They come from FactEHR, which builds on
MedAlign, and are covered by a Stanford Dataset DUA that does not allow redistribution.

`dataset/factehr/README.md` explains how to request access and rebuild the four input
files. You need them before you can generate anything:

| `--dataset` | File | Samples |
|---|---|---|
| `ehr_procedures` | `dataset/factehr/procedures.jsonl` | 200 |
| `ehr_progress_note` | `dataset/factehr/progress_note.jsonl` | 200 |
| `ehr_nursing_note` | `dataset/factehr/nursing_note.jsonl` | 100 |
| `ehr_discharge_summary` | `dataset/factehr/discharge_summary.jsonl` | 100 |

Each line holds a `prompt` (fact decomposition instruction plus the clinical note) and a
`natural_text` (the GPT-4o reference decomposition released with FactEHR).

## 3. SIR models (only for `--algorithm SIR`)

SIR needs a transform model and a sentence embedder, together about 1.3 GB. They are not
in the repository:

```bash
python download_sir_models.py
```

The token-to-partition mapping files in `watermark/sir/mapping/` **are** in the
repository and must be kept. `300_mapping_128256.json` is for Llama-3 based models and
`300_mapping_262208.json` is for Gemma based models. If a mapping file is missing, SIR
generates a new random one, and the results will not match the paper.

## 4. Generate

```bash
cd evaluation_scripts

python text_generation.py --algorithm KGW      --model llama3 --dataset ehr_nursing_note --gamma 0.5 --delta 2
python text_generation.py --algorithm SWEET    --model llama3 --dataset ehr_nursing_note --gamma 0.25 --delta 8 --entropy 0.9
python text_generation.py --algorithm DIP      --model llama3 --dataset ehr_nursing_note --alpha 0.45
python text_generation.py --algorithm Unbiased --model llama3 --dataset ehr_nursing_note --type gamma
python text_generation.py --algorithm SIR      --model llama3 --dataset ehr_nursing_note
```

`run_generation.sh` runs the full main-experiment sweep (5 algorithms x 4 tasks) and has
the model-ablation commands as comments.

Options:

- `--model`: `llama3` (main paper), `med42`, `medgemma`, `gemma` (model ablation)
- `--samples N`: stop after N prompts. Useful for a quick check.
- `--output FILE`: write a small JSON run summary.

All models are loaded in 4-bit NF4 quantization with bfloat16 compute, `temperature=0.7`,
`top_p=0.95`, `no_repeat_ngram_size=4`, `max_new_tokens=200`, `min_new_tokens=20`, seed
113. A generation shorter than `min_new_tokens` is retried up to 3 times.

## 5. Output

Results go to `logs/<ALGORITHM>/`:

- `<ALGORITHM>-<params>-<model>-<dataset>.pkl` — a tuple
  `(watermarked_texts, unwatermarked_texts, natural_texts)`, read by the evaluation
  scripts.
- `<ALGORITHM>-<params>-<model>-<dataset>.pkl.json` — the same data with the prompts, for
  reading by eye.

Example: `logs/KGW/KGW-g0.5-d2-llama3-ehr_nursing_note.pkl`.

## Note on the config files

`text_generation.py` writes the command line parameters back into `config/<ALGORITHM>.json`
before it loads the watermark, because the watermark reads its settings from that file.
Your working tree will therefore show a change in `config/` after a run with parameters
that differ from the defaults. This is expected.
