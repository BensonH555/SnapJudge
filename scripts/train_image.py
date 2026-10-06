import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import glob
import json
import math

import numpy as np
import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, DistributedSampler

from inference import load_split
from inference_image import Head, ImageDataset, build_processor, collate_fn, forward, load_backbone, load_model, score_images


class Wrap(torch.nn.Module):
    """Model and head in one module, so that DDP synchronises both."""

    def __init__(self, model, head):
        super().__init__()
        self.m, self.h = model, head

    def forward(self, enc):
        return forward(self.m, self.h, enc)


def macro_auroc(families, labels, scores):
    """Mean over generator families of AUROC(AI images of the family vs all human images)."""
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
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    grad_accum, rest = divmod(t["effective_batch_size"], t["per_device_train_batch_size"] * world_size)
    assert rest == 0, "effective_batch_size must be a multiple of per_device_train_batch_size x GPUs"
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    torch.manual_seed(t["seed"])

    processor, chat = build_processor(cfg)
    model = load_backbone(cfg).to(device)
    for name, p in model.named_parameters():
        if "visual" in name.lower() or "vision" in name.lower():
            p.requires_grad = False
    targets = [
        name for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and "visual" not in name.lower() and "vision" not in name.lower()
        and name.rsplit(".", 1)[-1] in lc["target_modules"]
    ]
    lora_config = LoraConfig(
        r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"], target_modules=targets, bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    head = Head(model.config.text_config.hidden_size).to(device)
    ddp = torch.nn.parallel.DistributedDataParallel(Wrap(model, head), device_ids=[rank], find_unused_parameters=True)

    data = ImageDataset(load_split(cfg["data"]["path"], cfg["data"]["config"], "train"))
    sampler = DistributedSampler(data, world_size, rank, shuffle=True, seed=t["seed"])
    loader = DataLoader(
        data,
        batch_size=t["per_device_train_batch_size"],
        sampler=sampler,
        num_workers=t["dataloader_num_workers"],
        collate_fn=collate_fn(processor, chat),
        pin_memory=True,
    )
    params = [p for p in ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(t["learning_rate"]), weight_decay=t["weight_decay"])
    total_steps = math.ceil(len(loader) / grad_accum) * t["num_train_epochs"]
    warmup = t["warmup_steps"]
    # linear warm-up times cosine decay to 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(s / warmup, 1.0) *
        (0.5 * (1 + math.cos(math.pi * min(s / max(total_steps, 1), 1.0)))))
    loss_fn = torch.nn.CrossEntropyLoss()

    step = 0
    for epoch in range(t["num_train_epochs"]):
        sampler.set_epoch(epoch)
        ddp.train()
        for i, enc in enumerate(loader):
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            loss = loss_fn(ddp(enc), enc["labels"]) / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and step % t["logging_steps"] == 0:
                    print(f"step {step}/{total_steps} loss {loss.item() * grad_accum:.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
        if rank == 0:
            checkpoint = os.path.join(t["output_dir"], f"epoch-{epoch + 1}")
            model.save_pretrained(checkpoint)
            torch.save(head.state_dict(), os.path.join(checkpoint, "head.pt"))
        dist.barrier()
    dist.destroy_process_group()
    return rank


def select_epoch(cfg):
    out = cfg["training"]["output_dir"]
    val = load_split(cfg["data"]["path"], cfg["data"]["config"], "validation")
    labels, families = list(val["label_id"]), list(val["family"])
    data = ImageDataset(val)
    results = []
    checkpoints = sorted(glob.glob(os.path.join(out, "epoch-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    for checkpoint in checkpoints:
        model, head, processor, chat = load_model(cfg, checkpoint)
        scores = score_images(model, head, processor, chat, data, cfg["inference"]["batch_size"])
        auroc = macro_auroc(families, labels, scores)
        results.append({"checkpoint": checkpoint, "val_auroc": auroc})
        print(results[-1], flush=True)
        del model, head
        gc.collect()
        torch.cuda.empty_cache()
    best = max(results, key=lambda r: r["val_auroc"])  # ties -> earlier epoch
    with open(os.path.join(out, "selection.json"), "w") as f:
        json.dump({"best_checkpoint": best["checkpoint"], "epochs": results}, f, indent=1)
    print(f"Selected {best['checkpoint']}")


def main():
    parser = argparse.ArgumentParser(description="Train the image detector SnapJudge")
    parser.add_argument("--config", required=True, help="configs/image.yaml")
    parser.add_argument("--dataset", default=None, help="Override data.path (e.g. a local copy of the dataset)")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.dataset:
        cfg["data"]["path"] = args.dataset

    if train(cfg) == 0:
        gc.collect()
        torch.cuda.empty_cache()
        select_epoch(cfg)


if __name__ == "__main__":
    main()
