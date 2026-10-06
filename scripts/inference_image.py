"""
Score images with the image detector SnapJudge.

Adds a column `score`: the probability that the passage shown in the image is AI-generated.

Usage:
  python scripts/inference_image.py \
    --config configs/image.yaml \
    --checkpoint SnapJudge/image \
    --dataset Benson555/SnapJudge \
    --split test \
    --output predictions.jsonl
"""

import argparse
import io
import os

import torch
import yaml
from datasets import Image as ImageFeature
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from inference import load_split, save

Image.MAX_IMAGE_PIXELS = None


class Head(torch.nn.Module):
    """LayerNorm + bias-free linear layer on the last token's hidden state."""

    def __init__(self, hidden_size):
        super().__init__()
        self.ln = torch.nn.LayerNorm(hidden_size)
        self.fc = torch.nn.Linear(hidden_size, 2, bias=False)

    def forward(self, h):
        return self.fc(self.ln(h))


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds.cast_column("image", ImageFeature(decode=False))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        image = row["image"]
        image = Image.open(io.BytesIO(image["bytes"]) if image["bytes"] else image["path"]).convert("RGB")
        return image, row.get("label_id", -1)


def build_processor(cfg):
    name = cfg["model"]["name"]
    revision = None if os.path.isdir(name) else cfg["model"]["revision"]
    processor = AutoProcessor.from_pretrained(
        name, revision=revision, min_pixels=cfg["data"]["min_pixels"], max_pixels=cfg["data"]["max_pixels"]
    )
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg["data"]["prompt"]}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor, chat


def collate_fn(processor, chat):
    def collate(batch):
        enc = processor(text=[chat] * len(batch), images=[b[0] for b in batch], padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b[1] for b in batch])
        return enc

    return collate


def load_backbone(cfg):
    name = cfg["model"]["name"]
    revision = None if os.path.isdir(name) else cfg["model"]["revision"]
    model = AutoModelForImageTextToText.from_pretrained(
        name, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    return model


def forward(model, head, enc):
    inputs = {k: v for k, v in enc.items() if k != "labels"}
    out = model(**inputs, output_hidden_states=True, logits_to_keep=1)
    h = out.hidden_states[-1]
    last = enc["attention_mask"].sum(1) - 1
    h = h[torch.arange(h.size(0), device=h.device), last]
    return head(h.float())


def load_model(cfg, checkpoint, device="cuda"):
    """checkpoint: directory with the LoRA adapter and head.pt."""
    processor, chat = build_processor(cfg)
    model = load_backbone(cfg).to(device)
    model = PeftModel.from_pretrained(model, checkpoint).eval()
    state = torch.load(os.path.join(checkpoint, "head.pt"), map_location="cpu")
    head = Head(state["fc.weight"].shape[1])
    head.load_state_dict(state)
    return model, head.to(device).eval(), processor, chat


@torch.no_grad()
def score_images(model, head, processor, chat, data, batch_size, device="cuda"):
    loader = DataLoader(
        data, batch_size=batch_size, num_workers=6, collate_fn=collate_fn(processor, chat), pin_memory=True
    )
    scores = []
    for enc in loader:
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        scores += forward(model, head, enc).softmax(-1)[:, 1].tolist()
    return scores


def main():
    parser = argparse.ArgumentParser(description="Score images with SnapJudge")
    parser.add_argument("--config", required=True, help="configs/image.yaml")
    parser.add_argument("--checkpoint", required=True, help="Trained adapter + head.pt (local directory)")
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset, local copy, or .parquet file with an image column")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, help="Output file (.jsonl or .csv)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = load_split(args.dataset or cfg["data"]["path"], cfg["data"]["config"], args.split)
    model, head, processor, chat = load_model(cfg, args.checkpoint)
    scores = score_images(model, head, processor, chat, ImageDataset(ds), cfg["inference"]["batch_size"])
    save(ds, scores, args.output)
    print(f"Scored {len(ds)} images. Saved to {args.output}")


if __name__ == "__main__":
    main()
