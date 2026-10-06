"""
Score passages with the LaTeX or text-based detector.

Adds a column `score`: the probability that the passage is AI-generated, averaged over
512-token windows (stride 256).

Usage:
  python scripts/inference.py \
    --config configs/latex.yaml \
    --checkpoint SnapJudge/latex \
    --dataset Benson555/SnapJudge \
    --split test \
    --output predictions.jsonl
"""

import argparse
import json
import os

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from preprocess import clean_text


class NormedLinear(torch.nn.Module):
    """LayerNorm + bias-free linear layer on the last token's hidden state."""

    def __init__(self, hidden_size, num_labels, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(self.norm(x))


def load_base_model(cfg, device_map=None):
    """Language model of Qwen3.5-27B with a new two-class head."""
    name = cfg["model"]["name"]
    revision = None if os.path.isdir(name) else cfg["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(name, revision=revision).text_config
    config.architectures = ["Qwen3_5TextForSequenceClassification"]
    config.num_labels = 2
    model, info = AutoModelForSequenceClassification.from_pretrained(
        name,
        config=config,
        revision=revision,
        dtype=torch.bfloat16,
        device_map=device_map,
        output_loading_info=True,
    )
    missing = [k for k in info["missing_keys"] if not k.startswith("score.")]
    if missing or info["mismatched_keys"]:
        raise RuntimeError(f"Backbone weights not loaded correctly: {missing[:5]}")
    model.config.pad_token_id = tokenizer.pad_token_id
    device = model.model.norm.weight.device
    model.score = NormedLinear(model.config.hidden_size, 2, device=device, dtype=torch.bfloat16)
    return model, tokenizer


def load_model(cfg, checkpoint):
    model, tokenizer = load_base_model(cfg, device_map={"": "cuda"})
    model = PeftModel.from_pretrained(model, checkpoint).eval()
    return model, tokenizer


def make_windows(tokenizer, text, window, stride):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= window:
        return [text]
    starts = range(0, len(ids) - window + stride, stride)
    return [tokenizer.decode(ids[s:s + window]) for s in starts]


@torch.no_grad()
def score_texts(model, tokenizer, texts, cfg):
    """Mean window probability per passage (None if the passage is empty)."""
    window, stride = cfg["data"]["window"], cfg["data"]["stride"]
    batch_size = cfg["inference"]["batch_size"]

    windows, owner = [], []
    for i, text in enumerate(texts):
        text = clean_text(text)
        if text:
            w = make_windows(tokenizer, text, window, stride)
            windows += w
            owner += [i] * len(w)

    device = next(model.parameters()).device
    probs = []
    for i in range(0, len(windows), batch_size):
        enc = tokenizer(
            windows[i:i + batch_size], truncation=True, max_length=window, padding=True, return_tensors="pt"
        ).to(device)
        probs += torch.softmax(model(**enc).logits.float(), dim=1)[:, 1].tolist()

    sums, counts = [0.0] * len(texts), [0] * len(texts)
    for i, p in zip(owner, probs):
        sums[i] += p
        counts[i] += 1
    return [s / c if c else None for s, c in zip(sums, counts)]


def load_split(path, name, split):
    """Hugging Face dataset (remote or local copy) or a .csv / .jsonl / .parquet file."""
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lstrip(".")
        return load_dataset("json" if ext == "jsonl" else ext, data_files=path, split="train")
    return load_dataset(path, name, split=split)


def save(ds, scores, path):
    ds = ds.remove_columns([c for c in ds.column_names if c == "image"]).add_column("score", scores)
    if path.endswith(".csv"):
        ds.to_pandas().to_csv(path, index=False)
    else:
        with open(path, "w") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Score passages with the LaTeX or text-based detector")
    parser.add_argument("--config", required=True, help="configs/latex.yaml or configs/text.yaml")
    parser.add_argument("--checkpoint", required=True, help="Trained adapter (local directory)")
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset, local copy, or .csv/.jsonl/.parquet file")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--output", required=True, help="Output file (.jsonl or .csv)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = load_split(args.dataset or cfg["data"]["path"], cfg["data"]["config"], args.split)
    model, tokenizer = load_model(cfg, args.checkpoint)
    scores = score_texts(model, tokenizer, list(ds[args.text_col]), cfg)
    save(ds, scores, args.output)
    print(f"Scored {len(ds)} passages. Saved to {args.output}")


if __name__ == "__main__":
    main()
