# Evaluating LLM Watermarking for Clinical EHR: Detectability, Style Shifts, and Helpfulness

Code for the EMNLP 2026 Main Conference paper.

This repository is a fork of [MarkLLM](https://github.com/THU-BPM/MarkLLM). The pipeline
has three stages: generate watermarked and unwatermarked text, measure how detectable the
watermark is, and measure what the watermark does to style and helpfulness.

**No experiment output is included.** No generated text, no judge output, no result
tables. Everything downstream of the clinical notes inherits their data use agreement, so
you regenerate it locally with the scripts here.

The upstream MarkLLM documentation, covering the toolkit itself and the watermarking
algorithms it implements, is kept as [README_MarkLLM.md](README_MarkLLM.md).

## What is in this repository

| Path | Purpose |
|---|---|
| `evaluation_scripts/text_generation.py` | stage 1, generate watermarked and unwatermarked text |
| `evaluation_scripts/detection_evaluation.py` | stage 2, TPR at a target FPR and AUROC |
| `evaluation_scripts/style_evaluation.py` | stage 3, per-sample style, accuracy and HELP |
| `evaluation_scripts/style_functions.py` | the metric definitions, including the HELP formula |
| `dataset/factehr/` | how to rebuild the input notes under your own DUA |
| `watermark/`, `config/` | the watermarking algorithms, from MarkLLM |

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

## 6. Detection

Scores each generated text with the matching watermark detector and reports TPR at a
target FPR together with AUROC. The watermarked text is compared against the natural
(human reference) text, which is truncated to the watermarked length first.

```bash
cd evaluation_scripts

python detection_evaluation.py --algorithm KGW --model llama3 --dataset ehr_nursing_note --gamma 0.5 --delta 2
```

`run_detection.sh` runs the full sweep. Pass the same algorithm parameters you used for
generation, so the detector matches the generator.

Detection scores are cached next to the generation output as `-W_SCORES.pkl`,
`-UW_SCORES.pkl`, `-N_SCORES.pkl` and `-SCORES.pkl`. Add `--metrics_only` to recompute the
metrics from that cache without loading the model again.

By default it reports both 0% and 5% FPR. `--target_fpr` picks a single operating point.
Score pairs where either the watermarked or the natural score is NaN are dropped from both
sides, so the two arrays stay aligned; the script prints how many it dropped.

## 7. Style and quality

Computes per-sample metrics for the watermarked and unwatermarked text against the natural
text:

```bash
python style_evaluation.py --task ehr_nursing_note
python style_evaluation.py --task ehr_procedures --algorithms KGW SWEET
```

Writes `results/style_<task>.csv`, one row per sample:

| Column | Meaning |
|---|---|
| `METEOR_U` / `METEOR_W` | accuracy against the reference decomposition |
| `BERTScore_F1_U` / `_W` | accuracy, reported for reference |
| `EMD_U_N` / `EMD_W_N` | Earth Mover's Distance over POS tag distributions, against the natural text |
| `GFI_U` / `GFI_W` | Gunning Fog Index |
| `FRE_U` / `FRE_W` | Flesch Reading Ease, reported for reference |
| `HELP_U` / `HELP_W` | fitted proxy for the judge Helpfulness score, on the 1-5 scale |
| `Delta_HELP` | `HELP_U - HELP_W`, the helpfulness lost to the watermark |

`_U` is unwatermarked, `_W` is watermarked. The natural text is truncated to 200 tokens to
match `max_new_tokens`, and `//` fact delimiters are replaced with periods before the
style metrics, so they do not distort sentence counts and POS tags.

## Note on the config files

`text_generation.py` writes the command line parameters back into `config/<ALGORITHM>.json`
before it loads the watermark, because the watermark reads its settings from that file.
Your working tree will therefore show a change in `config/` after a run with parameters
that differ from the defaults. This is expected.
