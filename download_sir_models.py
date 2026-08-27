#!/usr/bin/env python3
"""
Download the models required by the SIR watermarking algorithm.

Run this only if you want to reproduce the SIR results:

    python download_sir_models.py

It writes to watermark/sir/model/, which the repository does not track.
"""

import sys
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).parent
SIR_MODEL_DIR = PROJECT_ROOT / "watermark" / "sir" / "model"


def download_sir_models():
    print("=" * 80)
    print("DOWNLOADING SIR MODELS")
    print("=" * 80)

    SIR_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Transform model (transform_model_cbert.pth)
    print("\n1. Downloading transform model (transform_model_cbert.pth)...")
    print("-" * 80)

    transform_model_path = SIR_MODEL_DIR / "transform_model_cbert.pth"

    if transform_model_path.exists():
        print(f"Transform model already exists at {transform_model_path}")
    else:
        downloaded_file = hf_hub_download(
            repo_id="Generative-Watermark-Toolkits/MarkLLM-sir",
            filename="transform_model_cbert.pth",
            local_dir=SIR_MODEL_DIR
        )
        print(f"Downloaded transform model to {downloaded_file}")

    # 2. Sentence embedder (compositional-bert-large-uncased)
    print("\n2. Downloading compositional-bert-large-uncased...")
    print("-" * 80)

    embedder_path = SIR_MODEL_DIR / "compositional-bert-large-uncased"

    if (embedder_path / "config.json").exists():
        print(f"Compositional-BERT already exists at {embedder_path}")
    else:
        snapshot_download(
            repo_id="perceptiveshawty/compositional-bert-large-uncased",
            local_dir=embedder_path
        )
        print(f"Downloaded compositional-bert to {embedder_path}")

    # 3. Verify both models load
    print("\n3. Verifying models...")
    print("-" * 80)

    from watermark.sir.transform_model import TransformModel

    transform_model = TransformModel(input_dim=1024)
    transform_model.load_state_dict(torch.load(transform_model_path))
    print("Transform model loaded successfully")

    AutoTokenizer.from_pretrained(str(embedder_path))
    AutoModel.from_pretrained(str(embedder_path))
    print("Compositional-BERT loaded successfully")

    print("\n" + "=" * 80)
    print("ALL MODELS DOWNLOADED AND VERIFIED SUCCESSFULLY")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = download_sir_models()
    sys.exit(0 if success else 1)
