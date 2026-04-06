"""
train_asymmetric_biencoder.py
Training script for asymmetric bi-encoder retrieval.

Query encoder : BAAI/bge-base-zh-v1.5
Document side : pre-computed Qwen3-VL-Embedding-2B embeddings

Five variants (--variant):
  fullft     : full fine-tune BGE
  lora       : LoRA (r=8) on BGE attention + FFN
  query_mlp  : BGE frozen + MLPx4 or GLUx4 on query side
  doc_mlp    : BGE frozen + MLPx4 or GLUx4 on doc side
  both_mlp   : BGE frozen + projections on both sides

Example smoke test (no GPU contention, ~5 min):
  python src/train_asymmetric_biencoder.py \\
    --config config/asym_lora.yaml \\
    --smoke_test \\
    --gpu 0

Full training:
  accelerate launch --num_processes 2 src/train_asymmetric_biencoder.py \\
    --config config/asym_lora.yaml
"""

import os
import sys
import json
import time
import shutil
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import wandb
from accelerate import Accelerator
from tqdm import tqdm

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from asymmetric_biencoder_model import AsymmetricBiEncoderModel
from asymmetric_biencoder_dataset import AsymmetricBiEncoderDataset


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logger(log_path: str, name: str = "asym") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str, overrides: dict) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Apply CLI overrides (flat key=value, e.g. training.lr=3e-4)
    for k, v in overrides.items():
        keys = k.split(".")
        d = cfg
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = v
    return cfg


# ---------------------------------------------------------------------------
# Contrastive (InfoNCE) loss
# ---------------------------------------------------------------------------

def contrastive_loss(query_emb: torch.Tensor, doc_emb: torch.Tensor,
                     n_neg: int, negatives_x_device: bool,
                     accelerator: Accelerator) -> torch.Tensor:
    """
    InfoNCE loss matching query i to its positive at doc index i*(1+n_neg).
    doc_emb layout: [pos_0, neg_0_1..neg_0_N, pos_1, neg_1_1..neg_1_N, ...]

    Args:
        query_emb        : [B, 768]
        doc_emb          : [B*(1+n_neg), 768]
        n_neg            : number of negatives per query
        negatives_x_device : gather negatives across GPUs for larger batch
        accelerator      : HuggingFace Accelerator instance
    """
    if negatives_x_device:
        # Gather across all GPUs → larger effective batch of negatives
        query_emb = accelerator.gather(query_emb)
        doc_emb   = accelerator.gather(doc_emb)

    # Similarity matrix [B, B*(1+n_neg)]
    scores = torch.matmul(query_emb, doc_emb.T)

    B = query_emb.shape[0]
    # Label i: the positive of query i lives at position i*(1+n_neg)
    labels = torch.arange(B, device=scores.device, dtype=torch.long)
    labels = labels * (1 + n_neg)

    return F.cross_entropy(scores, labels)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: AsymmetricBiEncoderModel, optimizer,
                    scheduler, step: int, best_metric: float,
                    save_dir: str, suffix: str = ""):
    """Save model, optimizer, scheduler states and training metadata."""
    ckpt_dir = os.path.join(save_dir, f"checkpoint{suffix}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save(ckpt_dir)
    torch.save(optimizer.state_dict(), f"{ckpt_dir}/optimizer.pt")
    torch.save(scheduler.state_dict(), f"{ckpt_dir}/scheduler.pt")
    meta = {"step": step, "best_metric": best_metric}
    with open(f"{ckpt_dir}/train_meta.json", "w") as f:
        json.dump(meta, f)


def load_checkpoint(model: AsymmetricBiEncoderModel, optimizer, scheduler,
                    load_dir: str):
    """Load model + optimizer + scheduler from checkpoint dir. Returns (step, best_metric)."""
    ckpt_dir = load_dir
    if not os.path.isdir(ckpt_dir):
        return 0, 0.0

    # Model weights
    variant = model.variant
    if variant == "lora":
        from peft import PeftModel
        model.bge.load_adapter(ckpt_dir, adapter_name="default")
    else:
        bge_path = f"{ckpt_dir}/bge_state_dict.pt"
        if os.path.exists(bge_path):
            model.bge.load_state_dict(torch.load(bge_path, map_location="cpu"))
    for proj_name in ("query_proj", "doc_proj"):
        proj = getattr(model, proj_name, None)
        ppath = f"{ckpt_dir}/{proj_name}.pt"
        if proj is not None and os.path.exists(ppath):
            proj.load_state_dict(torch.load(ppath, map_location="cpu"))

    # Optimizer / scheduler
    opt_path = f"{ckpt_dir}/optimizer.pt"
    sch_path = f"{ckpt_dir}/scheduler.pt"
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
    if os.path.exists(sch_path):
        scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))

    meta_path = f"{ckpt_dir}/train_meta.json"
    step, best_metric = 0, 0.0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        step        = meta.get("step", 0)
        best_metric = meta.get("best_metric", 0.0)
    return step, best_metric


# ---------------------------------------------------------------------------
# Build optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model: AsymmetricBiEncoderModel, opt_cfg: dict):
    """Build optimizer from config. Defaults to AdamW."""
    name   = opt_cfg.get("name", "AdamW")
    kwargs = opt_cfg.get("kwargs", {})
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "AdamW":
        return torch.optim.AdamW(params, **kwargs)
    elif name == "Adam":
        return torch.optim.Adam(params, **kwargs)
    elif name == "Lamb":
        try:
            from torch_optimizer import Lamb
            return Lamb(params, **kwargs)
        except ImportError:
            raise ImportError("torch_optimizer not installed; use AdamW instead")
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, sch_cfg: dict, total_steps: int):
    """Build LR scheduler from config."""
    name   = sch_cfg.get("name", "LinearLR")
    kwargs = sch_cfg.get("kwargs", {})

    if name == "LinearLR":
        # Warm-up from 0 to lr over `warmup_steps`, then constant
        warmup = kwargs.get("warmup_steps", 0)
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3 if warmup > 0 else 1.0,
            end_factor=1.0,
            total_iters=max(warmup, 1),
        )
    elif name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict, args: argparse.Namespace):
    # ---- Timestamp-based run directory ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg['project_name']}_{cfg['model']['variant']}_{cfg['model']['proj_type']}_{ts}"
    project_dir = os.path.join(cfg["project_dir"], run_name)
    os.makedirs(project_dir, exist_ok=True)

    log_path = os.path.join(
        "/data/rech/huiyuche/TREC_iKAT_2024/logs",
        f"train_asym_{cfg['model']['variant']}_{cfg['model']['proj_type']}_{ts}.log"
    )
    logger = setup_logger(log_path)

    # ---- Accelerator (handles multi-GPU, mixed precision) ----
    accelerator = Accelerator(mixed_precision=cfg.get("mixed_precision", "no"))
    local_rank  = accelerator.local_process_index
    is_main     = accelerator.is_main_process

    if is_main:
        logger.info(f"Run name    : {run_name}")
        logger.info(f"Project dir : {project_dir}")
        logger.info(f"Log file    : {log_path}")
        logger.info(f"Variant     : {cfg['model']['variant']}")
        logger.info(f"Proj type   : {cfg['model']['proj_type']}")
        logger.info(f"Num GPUs    : {accelerator.num_processes}")
        logger.info(f"Config:\n{yaml.dump(cfg, default_flow_style=False)}")

    # ---- Wandb (main process only) ----
    if is_main:
        wandb.init(
            project=cfg.get("wandb_project", "qilin_multimodal_ir"),
            name=run_name,
            config=cfg,
        )

    # ---- Dataset & DataLoader ----
    train_cfg  = cfg["training"]
    dset_cfg   = cfg["datasets"]
    model_cfg  = cfg["model"]

    if is_main:
        logger.info("Building dataset …")

    dataset = AsymmetricBiEncoderDataset(cfg, split="train")

    loader = dataset.get_dataloader(
        tokenizer_path    = model_cfg["model_name_or_path"],
        max_length        = dset_cfg.get("max_length", 256),
        batch_size        = dset_cfg["batch_size"],
        shuffle           = True,
        num_workers       = dset_cfg.get("num_workers", 4),
        query_instruction = dset_cfg.get("query_instruction", ""),
    )

    # ---- Model ----
    if is_main:
        logger.info("Building model …")
    model = AsymmetricBiEncoderModel(cfg)

    # Print trainable param count
    if is_main:
        total   = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable params: {trainable:,} / {total:,} "
                    f"({100*trainable/total:.1f}%)")

    # ---- Optimizer & Scheduler ----
    num_epochs = train_cfg["num_epochs"]
    steps_per_epoch = len(loader)
    total_steps = num_epochs * steps_per_epoch

    optimizer = build_optimizer(model, cfg["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], total_steps)

    # ---- Resume from checkpoint ----
    resume_step = 0
    best_metric = 0.0
    if train_cfg.get("resume", False):
        ckpt_dir = os.path.join(project_dir, "checkpoint")
        if os.path.isdir(ckpt_dir):
            resume_step, best_metric = load_checkpoint(
                model, optimizer, scheduler, ckpt_dir
            )
            if is_main:
                logger.info(f"Resumed from step {resume_step}, best_metric={best_metric:.4f}")

    # ---- Accelerate prepare ----
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    # ---- Hyperparameter summary ----
    if is_main:
        logger.info("=" * 60)
        logger.info("Hyperparameter summary")
        logger.info(f"  variant          : {model_cfg['variant']}")
        logger.info(f"  proj_type        : {model_cfg['proj_type']}")
        logger.info(f"  doc_dim          : {model_cfg['doc_dim']}")
        logger.info(f"  batch_size       : {dset_cfg['batch_size']}")
        logger.info(f"  negative_samples : {dset_cfg['negative_samples']}")
        logger.info(f"  max_length       : {dset_cfg.get('max_length', 256)}")
        logger.info(f"  num_epochs       : {num_epochs}")
        logger.info(f"  total_steps      : {total_steps}")
        logger.info(f"  optimizer        : {cfg['optimizer']['name']}")
        logger.info(f"  lr               : {cfg['optimizer']['kwargs']['lr']}")
        logger.info(f"  scheduler        : {cfg['scheduler']['name']}")
        logger.info(f"  negatives_x_dev  : {train_cfg.get('negatives_x_device', False)}")
        logger.info(f"  eval_steps       : {train_cfg['eval_steps']}")
        logger.info(f"  save_steps       : {train_cfg['save_steps']}")
        logger.info("=" * 60)

    n_neg            = dset_cfg["negative_samples"]
    negatives_x_dev  = train_cfg.get("negatives_x_device", False)
    eval_steps       = train_cfg["eval_steps"]
    save_steps       = train_cfg["save_steps"]
    log_steps        = train_cfg.get("log_steps", 50)
    global_step      = resume_step

    # ---- Training loop ----
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0_epoch   = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}",
                    disable=not is_main)

        for step, batch in enumerate(pbar):
            # Skip steps already done (resume)
            if global_step < resume_step:
                global_step += 1
                continue

            optimizer.zero_grad()

            # ---- Forward pass ----
            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch.get("token_type_ids")
            doc_embs       = batch["doc_embs"]   # [B*(1+N), doc_dim]

            # Unwrap for models that need it (DistributedDataParallel wrapper)
            raw_model = accelerator.unwrap_model(model)
            query_emb, doc_emb = raw_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                doc_embs=doc_embs,
                token_type_ids=token_type_ids,
            )

            loss = contrastive_loss(
                query_emb, doc_emb, n_neg, negatives_x_dev, accelerator
            )

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            loss_val = loss.detach().float().item()
            epoch_loss += loss_val
            global_step += 1

            pbar.set_postfix({"loss": f"{loss_val:.4f}",
                              "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            # ---- Periodic logging ----
            if is_main and global_step % log_steps == 0:
                lr_now = scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch} | Step {global_step} | "
                            f"loss={loss_val:.4f} | lr={lr_now:.2e}")
                wandb.log({
                    "train/loss": loss_val,
                    "train/lr":   lr_now,
                    "train/step": global_step,
                    "train/epoch": epoch,
                }, step=global_step)

            # ---- Periodic evaluation ----
            if global_step % eval_steps == 0:
                # TODO: implement evaluation once all doc embeddings are ready
                # For now just log a placeholder
                if is_main:
                    logger.info(f"[Eval] Step {global_step} – eval not yet implemented "
                                f"(doc encoding still in progress)")
                    wandb.log({"eval/placeholder": 0.0}, step=global_step)

            # ---- Periodic checkpoint save ----
            if is_main and global_step % save_steps == 0:
                save_checkpoint(
                    accelerator.unwrap_model(model),
                    optimizer, scheduler,
                    global_step, best_metric,
                    project_dir, suffix=""
                )
                logger.info(f"Checkpoint saved at step {global_step}")

        # ---- End of epoch ----
        avg_loss = epoch_loss / max(len(loader), 1)
        elapsed  = time.time() - t0_epoch
        if is_main:
            logger.info(f"Epoch {epoch} done | avg_loss={avg_loss:.4f} | "
                        f"time={elapsed/60:.1f} min")
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch":      epoch,
            }, step=global_step)

    # ---- Final save ----
    if is_main:
        final_dir = os.path.join(
            "/data/rech/huiyuche/huggingface",
            f"asym_biencoder_{cfg['model']['variant']}_{cfg['model']['proj_type']}"
        )
        accelerator.unwrap_model(model).save(final_dir)
        logger.info(f"Final model saved to {final_dir}")
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train asymmetric bi-encoder")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--smoke_test", action="store_true",
                   help="Mini E2E run: 200 samples, 2 epochs, no eval")
    p.add_argument("--gpu", type=str, default=None,
                   help="CUDA_VISIBLE_DEVICES override (e.g. '0' or '0,1')")
    p.add_argument("--override", nargs="*", default=[],
                   help="Config overrides in key=value format (dot-separated keys)")
    return p.parse_args()


def main():
    args = parse_args()

    # GPU selection
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load config
    overrides = dict(kv.split("=", 1) for kv in (args.override or []))
    cfg = load_config(args.config, overrides)

    # Smoke test overrides
    if args.smoke_test:
        print("[smoke_test] Overriding config for quick E2E test …")
        cfg["training"]["num_epochs"]  = 2
        cfg["training"]["eval_steps"]  = 999999  # skip eval
        cfg["training"]["save_steps"]  = 999999  # skip intermediate saves
        cfg["training"]["log_steps"]   = 1
        cfg["datasets"]["batch_size"]  = 4
        cfg["datasets"]["num_workers"] = 0  # avoid multiprocess issues in debug
        # Limit to already-encoded notes
        cfg["datasets"]["max_encoded_note_idx"] = 100000
        cfg["project_name"] = cfg["project_name"] + "_smoke"

    train(cfg, args)


if __name__ == "__main__":
    main()
