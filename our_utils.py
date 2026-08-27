import os
import json
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from constants import PROJECT_ROOT
from utils.transformers_config import TransformersConfig
from evaluation.tools.text_editor import TruncatePromptTextEditor

# Global constants for paths
MODEL_PATHS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "med42": "m42-health/Llama3-Med42-8B",
    "medgemma": "google/medgemma-4b-it",
    "gemma": "google/gemma-3-4b-it"
}

DATASET_PATHS = {
    "ehr_procedures": os.path.join(PROJECT_ROOT, "dataset", "factehr", "procedures.jsonl"),
    "ehr_progress_note": os.path.join(PROJECT_ROOT, "dataset", "factehr", "progress_note.jsonl"),
    "ehr_nursing_note": os.path.join(PROJECT_ROOT, "dataset", "factehr", "nursing_note.jsonl"),
    "ehr_discharge_summary": os.path.join(PROJECT_ROOT, "dataset", "factehr", "discharge_summary.jsonl")
}

# Default device
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PromptFormatter:
    """Handles prompt formatting for instruction-tuned models."""

    @staticmethod
    def format_instruction_prompt(prompt, tokenizer):
        """
        Apply chat template for instruction-tuned models.

        Args:
            prompt: Raw instruction prompt text
            tokenizer: Model tokenizer with chat template support

        Returns:
            Formatted prompt string with chat template applied
        """
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        return formatted_prompt

    @staticmethod
    def extract_response_only(generated_ids, input_ids, tokenizer):
        """
        Extract only the generated response, excluding the input prompt.

        Args:
            generated_ids: Full generation output token IDs
            input_ids: Input prompt token IDs
            tokenizer: Model tokenizer

        Returns:
            Decoded response text (without input prompt)
        """
        response_ids = generated_ids[:, input_ids.shape[1]:]
        response_text = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
        return response_text


class ModelLoader:
    """Handles consistent model loading across the project."""

    @staticmethod
    def load_model(model_type, device=DEFAULT_DEVICE, max_new_tokens=200, min_new_tokens=20):
        """
        Load a model and its tokenizer with consistent parameters.

        Args:
            model_type: String identifier for model (llama3, med42, medgemma, gemma)
            device: Device to load model to
            max_new_tokens: Maximum number of tokens to generate
            min_new_tokens: Minimum New tokens to generate

        Returns:
            TransformersConfig object with loaded model and tokenizer
        """

        print(f"Using max_new_tokens={max_new_tokens}, min_new_tokens={min_new_tokens}")

        model_path = MODEL_PATHS.get(model_type)
        if not model_path:
            raise ValueError(f"Unknown model type: {model_type}")

        # All models use the same 4-bit quantization for a fair comparison
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if model_type == "llama3" or model_type == "med42":
            vocab_size = 128256
        elif model_type == "medgemma" or model_type == "gemma":
            vocab_size = 262208  # From config.json
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return TransformersConfig(
            model=AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa"
            ),
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            device=device,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,  # used for factehr to ensure at least some generation
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id
        )


class TextProcessor:
    """Handles text preprocessing consistently across the project."""

    @staticmethod
    def remove_prompt(text, prompt, model_type=None, tokenizer=None):
        """
        Remove prompt from generated text consistently.

        Args:
            text: The full text (prompt + generated)
            prompt: The prompt to remove
            model_type: The model type
            tokenizer: Optional tokenizer

        Returns:
            Text with prompt removed
        """
        text_editor = TruncatePromptTextEditor()
        return text_editor.edit(text, prompt)

    @staticmethod
    def truncate_text(text, tokenizer, max_length=50):
        """
        Truncate text to a specified token length.

        Args:
            text: Text to truncate
            tokenizer: Tokenizer to use
            max_length: Maximum token length

        Returns:
            Truncated text
        """
        tokens = tokenizer(text, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length)['input_ids']
        return tokenizer.decode(tokens[0], skip_special_tokens=True)


class ConfigManager:
    """Manages algorithm configurations consistently."""

    @staticmethod
    def load_config(algorithm, params=None):
        """
        Load and update configuration from file.

        Args:
            algorithm: Algorithm name (e.g., "SWEET", "KGW")
            params: Dictionary of parameters to update

        Returns:
            Updated configuration dictionary
        """
        config_path = os.path.join(PROJECT_ROOT, "config", f"{algorithm}.json")
        with open(config_path) as f:
            config = json.load(f)

        # Update configuration with provided parameters
        if params:
            for key, value in params.items():
                if value is not None:
                    config[key] = value

        return config

    @staticmethod
    def save_config(algorithm, config):
        """
        Save configuration to file.

        Args:
            algorithm: Algorithm name
            config: Configuration dictionary
        """
        config_path = os.path.join(PROJECT_ROOT, "config", f"{algorithm}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)


class FileManager:
    """Handles file operations consistently."""

    @staticmethod
    def get_dataset_path(dataset):
        """Get dataset path from identifier."""
        normalized_dataset = dataset.lower()
        return DATASET_PATHS.get(normalized_dataset)

    @staticmethod
    def get_model_path(model):
        """Get model path from identifier."""
        return MODEL_PATHS.get(model.lower())

    @staticmethod
    def build_filename(algorithm, params, model, dataset):
        """
        Build consistent filename based on parameters.

        Args:
            algorithm: Algorithm name
            params: Algorithm parameters
            model: Model identifier
            dataset: Dataset identifier

        Returns:
            Constructed filename
        """
        def format_param(value):
            """Format parameter value to remove unnecessary decimal places"""
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)

        if algorithm == "KGW":
            return f"{algorithm}-g{format_param(params['gamma'])}-d{format_param(params['delta'])}-{model}-{dataset}.pkl"
        elif algorithm == "SWEET":
            return f"{algorithm}-e{format_param(params['entropy'])}-g{format_param(params['gamma'])}-d{format_param(params['delta'])}-{model}-{dataset}.pkl"
        elif algorithm == "DIP":
            return f"{algorithm}-a{format_param(params['alpha'])}-{model}-{dataset}.pkl"
        elif algorithm == "Unbiased":
            return f"{algorithm}-t{format_param(params['type'])}-{model}-{dataset}.pkl"
        elif algorithm == "SIR":
            return f"{algorithm}-{model}-{dataset}.pkl"
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    @staticmethod
    def parse_filename(filename):
        """
        Parse parameters from filename.

        Args:
            filename: Filename to parse

        Returns:
            Dictionary of extracted parameters
        """
        parts = filename.split('-')
        if "KGW" in parts:
            return {
                'algorithm': parts[0],
                'gamma': float(parts[1][1:]),
                'delta': float(parts[2][1:]),
                'model': parts[3],
                'dataset': parts[4].split('.')[0]
            }
        elif "SWEET" in parts:
            return {
                'algorithm': parts[0],
                'entropy': float(parts[1][1:]),
                'gamma': float(parts[2][1:]),
                'delta': float(parts[3][1:]),
                'model': parts[4],
                'dataset': parts[5].split('.')[0]
            }
        elif "DIP" in parts:
            return {
                'algorithm': parts[0],
                'alpha': float(parts[1][1:]),
                'model': parts[2],
                'dataset': parts[3].split('.')[0]
            }
        elif "Unbiased" in parts:
            return {
                'algorithm': parts[0],
                'type': parts[1],
                'model': parts[2],
                'dataset': parts[3].split('.')[0]
            }
        elif "SIR" in parts:
            return {
                'algorithm': parts[0],
                'model': parts[1],
                'dataset': parts[2].split('.')[0]
            }


# SOURCE: https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#F1
# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def extract_prompt_ehr(prompt_text):
    """Extract the actual note from EHR prompt."""
    delimiter = 'Do not include any other text, or say "Here is the list..." \n\nNote: \n'
    if delimiter in prompt_text:
        return prompt_text.split(delimiter, 1)[1].strip()
    return prompt_text
