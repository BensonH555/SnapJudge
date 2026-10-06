import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import glob
import json

import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from inference import load_base_model, load_model, load_split, make_windows, score_texts
from preprocess import clean_text


def load_passages(cfg, split):
    ds = load_split(cfg["data"]["path"], cfg["data"]["config"], split)
    rows = [r for r in zip(ds["text"], ds["label_id"], ds["family"]) if r[0]]
    texts, labels, families = map(list, zip(*rows))
    return texts, labels, families


def make_window_dataset(texts, labels, tokenizer, cfg):
    window, stride = cfg["data"]["window"], cfg["data"]["stride"]
    w_texts, w_labels = [], []
    for text, label in zip(texts, labels):
        w = make_windows(tokenizer, clean_text(text), window, stride)
        w_texts += w
        w_labels += [label] * len(w)
    ds = Dataset.from_dict({"text": w_texts, "labels": w_labels})
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=window), batched=True, remove_columns=["text"]
    )


def macro_auroc(families, labels, scores):
    """Mean over generator families of AUROC(AI passages of the family vs all human passages)."""
    families, labels, scores = np.asarray(families), np.asarray(labels), np.asarray(scores, dtype=float)
    human = scores[labels == 0]
    aurocs = []
    for family in sorted(set(families[labels == 1])):
        ai = scores[(labels == 1) & (families == family)]
        y = np.r_[np.zeros(len(human)), np.ones(len(ai))]
        aurocs.append(roc_auc_score(y, np.r_[human, ai]))
    return float(np.mean(aurocs))


def train(cfg):
    t, lc = cfg["training"], cfg["lora"]
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum, rest = divmod(t["effective_batch_size"], t["per_device_train_batch_size"] * world_size)
    assert rest == 0, "effective_batch_size must be a multiple of per_device_train_batch_size x GPUs"
    torch.manual_seed(t["seed"])

    texts, labels, _ = load_passages(cfg, "train")
    model, tokenizer = load_base_model(cfg)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        target_modules=lc["target_modules"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    train_ds = make_window_dataset(texts, labels, tokenizer, cfg)

    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        learning_rate=float(t["learning_rate"]),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=t["num_train_epochs"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        save_strategy="epoch",
        logging_steps=t["logging_steps"],
        bf16=True,
        seed=t["seed"],
        data_seed=t["seed"],
        dataloader_num_workers=t["dataloader_num_workers"],
        ddp_find_unused_parameters=False,
        label_names=["labels"],
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    return trainer


def select_epoch(cfg):
    out = cfg["training"]["output_dir"]
    texts, labels, families = load_passages(cfg, "validation")
    results = []
    checkpoints = sorted(glob.glob(os.path.join(out, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    for checkpoint in checkpoints:
        model, tokenizer = load_model(cfg, checkpoint)
        scores = score_texts(model, tokenizer, texts, cfg)
        results.append({"checkpoint": checkpoint, "val_auroc": macro_auroc(families, labels, scores)})
        print(results[-1], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    best = max(results, key=lambda r: r["val_auroc"])  # ties -> earlier epoch
    with open(os.path.join(out, "selection.json"), "w") as f:
        json.dump({"best_checkpoint": best["checkpoint"], "epochs": results}, f, indent=1)
    print(f"Selected {best['checkpoint']}")


def main():
    parser = argparse.ArgumentParser(description="Train the LaTeX or text-based detector")
    parser.add_argument("--config", required=True, help="configs/latex.yaml or configs/text.yaml")
    parser.add_argument("--dataset", default=None, help="Override data.path (e.g. a local copy of the dataset)")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.dataset:
        cfg["data"]["path"] = args.dataset

    trainer = train(cfg)
    if trainer.is_world_process_zero():
        trainer.accelerator.free_memory()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        select_epoch(cfg)


if __name__ == "__main__":
    main()
