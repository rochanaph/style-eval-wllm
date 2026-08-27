# FactEHR data

The clinical notes used in this paper are **not in this repository**. They come from
FactEHR, which builds on MedAlign (`stanford-medalign-1.0`), and are covered by a
Stanford Dataset DUA that does not allow redistribution. This directory holds only the
script that rebuilds the input files once you have your own copy of the data, plus
`doc_ids.json`, which lists the exact 600 notes the paper used so everyone works on the
same subset. `doc_ids.json` holds identifiers only, no clinical text.

## What the generation scripts expect

Four JSONL files in this directory:

| File | Note type | Lines |
|---|---|---|
| `procedures.jsonl` | procedures | 200 |
| `progress_note.jsonl` | progress note | 200 |
| `nursing_note.jsonl` | nursing note | 100 |
| `discharge_summary.jsonl` | discharge summary | 100 |

Each line has two fields:

```json
{
  "prompt": "Please breakdown the following text into independent facts ... Note: \n<clinical note>",
  "natural_text": "There is a consolidation. // The consolidation is dense. // ..."
}
```

- `prompt` is the fact decomposition instruction with the clinical note appended. The
  watermarked model answers this prompt, and its answer is the text we watermark and
  evaluate.
- `natural_text` is the **GPT-4o** reference decomposition released with FactEHR. It is
  the human-written-style comparison text in the style evaluation.

## How to get the data

### 1. Request access to FactEHR

FactEHR is distributed through Stanford Redivis:

- Dataset: https://stanford.redivis.com/datasets/bckk-15p0mwmz7
- Paper: https://arxiv.org/abs/2412.12422
- Code: https://github.com/som-shahlab/factehr

You need to accept the Stanford Dataset DUA. Note the usage restrictions: the notes must
not be sent to commercial LLM APIs unless the service is HIPAA compliant and explicitly
permitted (Azure OpenAI, Amazon Bedrock, Google Vertex AI, Anthropic Claude).

The release gives you the combined notes CSV (`combined_notes_*.csv`, with `doc_id`,
`note_text`, `note_type`, `dataset_name`) and the fact decompositions CSV
(`fact_decompositions_*.csv`, with `doc_id`, `model`, `decomposition_result`).

### 2. Build the prompted dataset

Clone the FactEHR repository and convert the notes into a spaCy DocBin:

```python
import pandas as pd, spacy, os
from spacy.tokens import DocBin

df = pd.read_csv("combined_notes_110424.csv")
nlp = spacy.blank("en")
doc_bin = DocBin(store_user_data=True)

for _, row in df.iterrows():
    doc = nlp(row["note_text"])
    doc.user_data["doc_id"] = row["doc_id"]
    doc.user_data["note_type"] = row["note_type"]
    doc.user_data["dataset_name"] = row["dataset_name"]
    doc_bin.add(doc)

os.makedirs("data/datasets", exist_ok=True)
with open("data/datasets/factehr_combined_notes.docbin", "wb") as f:
    f.write(doc_bin.to_bytes())
```

Then build the prompted dataset. Use the **delimiter** prompt directory. The plain
`fact_decomposition/` templates make the model produce a numbered list, but this paper
needs the `//` separated format:

```bash
python scripts/build_fact_decomp_prompted_dataset.py \
  --path_to_input data/datasets/factehr_combined_notes.docbin \
  --path_to_prompt_dir data/prompt_templates/fact_decomposition_delimiter/ \
  --path_to_output_dir data/datasets/prompted/ \
  --file_name_prefix fact_decomposition_delimiter \
  --completion_format messages
```

This writes `data/datasets/prompted/fact_decomposition_delimiter_<date>.jsonl`, where
each line carries a `metadata` object with `doc_id` and `note_type`, and a `messages`
list whose first entry holds the prompt.

The prompt template is `fact_decomposition_delimiter/prompt1_icl.tmpl`:

```
Please breakdown the following text into independent facts as a string delimited by "//" to separate the facts

Example 1 :
Note: "There is a dense consolidation in the left lower lobe."

Atomic facts:
There is a consolidation. // The consolidation is dense. // The consolidation is on the left. // The consolidation is in a lobe. // The consolidation is in the lower portion of the left lobe.


Example 2:

Note: "The patient has been having intermittent shortness of breath for the last two years."

Atomic facts:
The patient has been having shortness of breath. // The shortness of breath is intermittent. // The shortness of breath has been present for the last two years.

Do not include any other text, or say "Here is the list..."

Note:
{text}
```

### 3. Build the four JSONL files

```bash
python build_factehr_dataset.py \
  --prompted_dataset /path/to/fact_decomposition_delimiter_<date>.jsonl \
  --decompositions /path/to/fact_decompositions_110424.csv
```

For each note type the script reads the `doc_id` list from `doc_ids.json`, takes the
prompt from the prompted dataset and the GPT-4o `decomposition_result` from the CSV, and
writes the four JSONL files into this directory. It reports any `doc_id` it cannot find
in either input.

Your decompositions CSV needs the columns `doc_id`, `model` and `decomposition_result`.

### 4. Check

```bash
wc -l *.jsonl
```

Expect 200, 200, 100 and 100 lines. Then run the generation as described in
`../../README_REPRODUCE.md`.

## Note on sample counts

The paper uses 200 procedures and progress notes but only 100 nursing notes and discharge
summaries, because FactEHR holds fewer notes of those two types (258 and 234 in total,
against roughly 500 each for the other two). The generation script reads at most 200
lines per file, set by `max_samples` in `FactEHRDataset`.

## If the rebuild does not match

The prompt string must match character for character, because it is the model input. Two
things usually go wrong:

- The wrong prompt directory. `fact_decomposition/` gives a numbered list;
  this paper needs `fact_decomposition_delimiter/`, which gives the `//` format.
- A newer FactEHR release with different `doc_id` values. In that case the script reports
  the missing ids. You can still run everything, but the notes will not be the same ones
  used in the paper.
